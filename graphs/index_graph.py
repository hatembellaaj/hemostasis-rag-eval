from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

from services.storage import Settings, ensure_dirs, new_id, write_json, write_jsonl, read_jsonl, now_iso
from services.chunking import make_chunks, SplitMode, ChunkMode
from services.metrics import chunk_stats


class IndexState(TypedDict, total=False):
    settings: Settings

    # inputs
    pivot_path: str
    split_mode: SplitMode
    chunk_mode: ChunkMode
    chunk_size: int
    chunk_overlap: int

    # intermediate
    pivot_rows: List[Dict[str, Any]]
    docs: List[Document]
    chunks: List[Document]

    # outputs
    index_run_id: str
    chunks_path: str
    stats: Dict[str, Any]
    chroma_count: Optional[int]


def _node_init(state: IndexState) -> IndexState:
    settings = state["settings"]
    ensure_dirs(settings)
    run_id = new_id("indexrun")
    return {
        **state,
        "index_run_id": run_id,
        "stats": {"created_at": now_iso(), "index_run_id": run_id},
    }


def _node_load_pivot(state: IndexState) -> IndexState:
    rows = read_jsonl(Path(state["pivot_path"]))
    return {**state, "pivot_rows": rows}


def _sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma requires metadata values to be only: str/int/float/bool.
    - drop None
    - convert dict/list to JSON string
    - keep only simple scalars
    """
    import json

    clean: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (dict, list, tuple)):
            # store as JSON string (still auditable)
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            # fallback: string
            clean[k] = str(v)
    return clean


def _node_rows_to_docs(state: IndexState) -> IndexState:
    docs: List[Document] = []
    for r in state.get("pivot_rows") or []:
        text = r.get("text", "") or ""

        base_md = {
            # flatten metadata for retrieval filters
            **(r.get("metadata") or {}),
            "source_type": r.get("source_type"),
            "source": r.get("source"),
            "doc_id": r.get("doc_id"),
            "snapshot_id": r.get("snapshot_id"),
        }

        # provenance is useful but must be scalar for Chroma
        prov = r.get("provenance") or {}
        base_md["provenance"] = prov  # will be JSON-stringified by sanitizer

        # IMPORTANT: sanitize for Chroma
        md = _sanitize_metadata(base_md)

        docs.append(Document(page_content=text, metadata=md))

    return {**state, "docs": docs}



def _node_chunk(state: IndexState) -> IndexState:
    chunks = make_chunks(
        docs=state.get("docs") or [],
        split_mode=state["split_mode"],
        chunk_mode=state["chunk_mode"],
        chunk_size=int(state["chunk_size"]),
        chunk_overlap=int(state["chunk_overlap"]),
        do_clean=True,
    )
    st = dict(state["stats"])
    st.update(chunk_stats(chunks))
    return {**state, "chunks": chunks, "stats": st}


def _node_persist_chunks(state: IndexState) -> IndexState:
    settings = state["settings"]
    run_id = state["index_run_id"]

    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(state.get("chunks") or []):
        rows.append(
            {
                "chunk_id": f"{c.metadata.get('doc_id')}::chunk::{i}",
                "text": c.page_content,
                "metadata": c.metadata,
            }
        )

    chunks_path = settings.artifacts_dir / "chunks" / f"{run_id}.jsonl"
    write_jsonl(chunks_path, rows)

    stats_path = settings.artifacts_dir / "stats" / f"{run_id}.json"
    write_json(stats_path, state.get("stats") or {})

    return {**state, "chunks_path": str(chunks_path)}


def _node_index_chroma(state: IndexState) -> IndexState:
    settings = state["settings"]

    if not (Path(settings.chroma_dir).exists()):
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    vs = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    # Add documents
    chunks = state.get("chunks") or []
    vs.add_documents(chunks)

    # Count (robust)
    try:
        count = vs._collection.count()
    except Exception:
        try:
            count = len(vs.get()["ids"])
        except Exception:
            count = None

    return {**state, "chroma_count": count}


def build_index_graph() -> Any:
    g = StateGraph(IndexState)

    g.add_node("init", _node_init)
    g.add_node("load_pivot", _node_load_pivot)
    g.add_node("rows_to_docs", _node_rows_to_docs)
    g.add_node("chunk", _node_chunk)
    g.add_node("persist_chunks", _node_persist_chunks)
    g.add_node("index_chroma", _node_index_chroma)

    g.set_entry_point("init")
    g.add_edge("init", "load_pivot")
    g.add_edge("load_pivot", "rows_to_docs")
    g.add_edge("rows_to_docs", "chunk")
    g.add_edge("chunk", "persist_chunks")
    g.add_edge("persist_chunks", "index_chroma")
    g.add_edge("index_chroma", END)

    return g.compile()
