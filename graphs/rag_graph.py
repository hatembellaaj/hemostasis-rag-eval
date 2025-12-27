from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from services.bm25_index import build_or_load_bm25_index
from services.storage import Settings, ensure_dirs, new_id, now_iso, write_json
from services.retrieval import (
    DenseRetriever,
    SparseBM25Retriever,
    build_filter,
    hybrid_retrieve,
    load_chunk_documents,
    retrieve,
)
from services.rerank import similarity_with_scores, apply_threshold
from services.prompts import SYSTEM_RAG_STRICT, PROTOCOL_TEMPLATE, build_user_prompt
from services.verify import rule_based_verify
from services.metrics import context_cost


class RAGState(TypedDict, total=False):
    settings: Settings

    # inputs
    question: str
    country: Optional[str]     # Tunisia or None
    disease: Optional[str]     # "Hemophilia" / "VWD" or None

    search_type: str           # "similarity" | "mmr"
    retrieval_mode: str        # "dense" | "sparse" | "hybrid"
    k: int
    fetch_k: int
    rerank: bool
    score_threshold: Optional[float]
    max_context_chars: int
    max_attempts: int

    # intermediate
    attempt: int
    candidates: List[Document]
    scored: List[Dict[str, Any]]
    context: str
    answer: str
    verification: Dict[str, Any]
    stop_reason: str

    # outputs
    run_id: str


def _node_init(state: RAGState) -> RAGState:
    settings = state["settings"]
    ensure_dirs(settings)
    run_id = new_id("ragrun")
    return {
        **state,
        "run_id": run_id,
        "attempt": 1,
    }


def _get_vs(settings: Settings) -> Chroma:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    return Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


@lru_cache(maxsize=4)
def _get_bm25_index(index_path: str, chunks_glob: str, text_field: str) -> Any:
    return build_or_load_bm25_index(index_path=index_path, chunks_glob=chunks_glob, text_field=text_field)


@lru_cache(maxsize=4)
def _get_chunk_lookup(chunks_glob: str, text_field: str) -> Dict[str, Document]:
    return load_chunk_documents(chunks_glob=chunks_glob, text_field=text_field)


def _node_retrieve(state: RAGState) -> RAGState:
    vs = _get_vs(state["settings"])
    retrieval_mode = state.get("retrieval_mode", "dense")
    text_field = state.get("text_field", "text")
    where = None

    # Single-country setting: if country=Tunisia, we filter only registry evidence by country
    # Literature is generally country=None; filtering literature by country would drop it.
    # We therefore apply filter only when disease is known or study_type is known.
    # Minimal practical filter: disease if provided + study_type optional.
    # You can later add "evidence_scope" to differentiate local/global.
    if state.get("disease"):
        where = {"disease": state["disease"]}

    # dense-only path retains existing behaviour
    if retrieval_mode == "dense":
        docs = retrieve(
            vs=vs,
            query=state["question"],
            search_type=state["search_type"],  # type: ignore
            k=int(state["k"]),
            fetch_k=int(state["fetch_k"]),
            where=where,
        )
        return {**state, "candidates": docs}

    chunks_glob = str(state["settings"].artifacts_dir / "chunks" / "*.jsonl")
    index_path = str(state["settings"].artifacts_dir / "bm25_index.pkl")
    sparse_retriever = SparseBM25Retriever(
        bm25_index=_get_bm25_index(index_path=index_path, chunks_glob=chunks_glob, text_field=text_field)
    )
    chunk_lookup = _get_chunk_lookup(chunks_glob=chunks_glob, text_field=text_field)

    if retrieval_mode == "sparse":
        results = sparse_retriever.retrieve(state["question"], top_k=int(state["k"]))
        docs: List[Document] = []
        for r in results:
            doc = chunk_lookup.get(r.get("chunk_id"))
            if not doc:
                continue
            md = dict(doc.metadata or {})
            md["score_sparse"] = r.get("score_sparse")
            md["chunk_id"] = md.get("chunk_id") or r.get("chunk_id")
            docs.append(Document(page_content=doc.page_content, metadata=md))
        return {**state, "candidates": docs}

    if retrieval_mode == "hybrid":
        dense_retriever = DenseRetriever(vs=vs, search_type=state.get("search_type", "similarity"))
        hybrid_results = hybrid_retrieve(
            query=state["question"],
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            k_dense=int(state["fetch_k"]),
            k_sparse=int(state["fetch_k"]),
            k_final=int(state["k"]),
            alpha=0.5,
        )

        docs: List[Document] = []
        for r in hybrid_results:
            doc = r.get("doc") or chunk_lookup.get(r.get("chunk_id"))
            if not doc:
                continue
            md = dict(doc.metadata or {})
            md["score_hybrid"] = r.get("score_hybrid")
            md["score_dense"] = r.get("score_dense")
            md["score_sparse"] = r.get("score_sparse")
            md["chunk_id"] = md.get("chunk_id") or r.get("chunk_id")
            docs.append(Document(page_content=doc.page_content, metadata=md))
        return {**state, "candidates": docs}

    # fallback to dense if an unknown mode is provided
    docs = retrieve(
        vs=vs,
        query=state["question"],
        search_type=state["search_type"],  # type: ignore
        k=int(state["k"]),
        fetch_k=int(state["fetch_k"]),
        where=where,
    )
    return {**state, "candidates": docs}


def _node_rerank_and_threshold(state: RAGState) -> RAGState:
    if not state.get("rerank"):
        # keep as-is, assign dummy scores
        scored = [{"rank": i + 1, "score": None, "doc": d} for i, d in enumerate(state.get("candidates") or [])]
        return {**state, "scored": scored}

    vs = _get_vs(state["settings"])
    where = {"disease": state["disease"]} if state.get("disease") else None

    # fetch with scores; use fetch_k to have enough for thresholding
    docs_scores = similarity_with_scores(vs, state["question"], k=int(state["fetch_k"]), where=where)
    docs_scores = apply_threshold(docs_scores, state.get("score_threshold"))

    # take top-k after threshold
    docs_scores = docs_scores[: int(state["k"])]

    scored = []
    for i, (d, s) in enumerate(docs_scores, start=1):
        scored.append({"rank": i, "score": float(s), "doc": d})

    # if threshold removes everything -> empty context scenario
    return {**state, "scored": scored}


def _node_build_context(state: RAGState) -> RAGState:
    max_chars = int(state["max_context_chars"])
    parts: List[str] = []
    used = 0
    for item in state.get("scored") or []:
        d: Document = item["doc"]
        rank = item["rank"]
        tag = f"[S{rank}]"
        md = d.metadata or {}
        prov = md.get("provenance", {})
        header = f"{tag} source={md.get('source')} type={md.get('source_type')} prov={prov}"
        body = (d.page_content or "").strip()

        block = header + "\n" + body
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2

    context = "\n\n".join(parts)
    st_cost = context_cost(context)

    return {**state, "context": context, **st_cost}


def _node_generate(state: RAGState) -> RAGState:
    llm = ChatOpenAI(model=state["settings"].chat_model, temperature=0.1)

    user = build_user_prompt(
        question=state["question"],
        country=state.get("country"),
        disease=state.get("disease"),
    )

    prompt = f"{SYSTEM_RAG_STRICT}\n\n{PROTOCOL_TEMPLATE}\n\nCONTEXT:\n{state.get('context','')}\n\nUSER:\n{user}\n"

    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {**state, "answer": answer}


def _node_verify(state: RAGState) -> RAGState:
    v = rule_based_verify(state.get("answer") or "")
    return {**state, "verification": v}


def _decide_next(state: RAGState) -> str:
    # stop if supported, else retry until max_attempts
    verdict = (state.get("verification") or {}).get("verdict", "UNSUPPORTED")
    if verdict == "SUPPORTED":
        return "stop_supported"

    attempt = int(state.get("attempt", 1))
    if attempt >= int(state.get("max_attempts", 1)):
        return "stop_max_attempts"

    # if no context -> no point retry unless we change fetch_k or threshold in future
    if not (state.get("context") or "").strip():
        return "stop_no_context"

    return "retry"


def _node_retry(state: RAGState) -> RAGState:
    # Simple retry policy for now: increase fetch_k slightly on each attempt
    attempt = int(state.get("attempt", 1)) + 1
    fetch_k = int(state.get("fetch_k", 10)) + 5
    return {**state, "attempt": attempt, "fetch_k": fetch_k}


def _node_persist_run(state: RAGState) -> RAGState:
    settings = state["settings"]
    run_id = state["run_id"]

    out = {
        "run_id": run_id,
        "created_at": now_iso(),
        "question": state.get("question"),
        "country": state.get("country"),
        "disease": state.get("disease"),
        "retrieval_mode": state.get("retrieval_mode"),
        "search_type": state.get("search_type"),
        "k": state.get("k"),
        "fetch_k": state.get("fetch_k"),
        "rerank": state.get("rerank"),
        "score_threshold": state.get("score_threshold"),
        "max_context_chars": state.get("max_context_chars"),
        "attempt": state.get("attempt"),
        "verification": state.get("verification"),
        "context_chars": state.get("context_chars"),
        "answer": state.get("answer"),
    }

    path = settings.artifacts_dir / "runs" / f"{run_id}.json"
    write_json(path, out)
    return state


def build_rag_graph() -> Any:
    g = StateGraph(RAGState)

    g.add_node("init", _node_init)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("rerank", _node_rerank_and_threshold)
    g.add_node("context", _node_build_context)
    g.add_node("generate", _node_generate)
    g.add_node("verify", _node_verify)
    g.add_node("retry", _node_retry)
    g.add_node("persist", _node_persist_run)

    g.set_entry_point("init")
    g.add_edge("init", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "context")
    g.add_edge("context", "generate")
    g.add_edge("generate", "verify")

    g.add_conditional_edges(
        "verify",
        _decide_next,
        {
            "stop_supported": "persist",
            "stop_max_attempts": "persist",
            "stop_no_context": "persist",
            "retry": "retry",
        },
    )
    g.add_edge("retry", "retrieve")
    g.add_edge("persist", END)

    return g.compile()
