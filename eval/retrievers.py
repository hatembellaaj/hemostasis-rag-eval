from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import maximal_marginal_relevance

from services.bm25_index import BM25Index, build_or_load_bm25_index
from services.retrieval import metadata_to_chunk_id
from services.storage import Settings

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")


@dataclass
class RetrievalResult:
    doc: Document
    score: Optional[float]
    chunk_id: Optional[str]
    score_dense: Optional[float] = None
    score_sparse: Optional[float] = None
    score_hybrid: Optional[float] = None


class RetrieverInterface(Protocol):
    def retrieve(self, query: str, k: int, fetch_k: int) -> List[RetrievalResult]:
        ...


def _load_chunks(chunks_glob: str) -> list[dict]:
    chunks: list[dict] = []
    for path_str in sorted(glob.glob(chunks_glob)):
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    continue
    return chunks


def _load_chunk_documents(chunks: list[dict]) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    for chunk in chunks:
        cid = str(chunk.get("chunk_id")) if chunk.get("chunk_id") is not None else None
        if not cid:
            continue
        text = chunk.get("text") or chunk.get("text_block")
        if not text:
            continue
        md = dict(chunk.get("metadata") or {})
        md["chunk_id"] = cid
        docs[cid] = Document(page_content=str(text), metadata=md)
    return docs


def _normalize_scores(results: list[RetrievalResult], key: str) -> dict[str, float]:
    scores = [getattr(r, key) for r in results if isinstance(getattr(r, key), (int, float))]
    if not scores:
        return {}
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return {r.chunk_id: 1.0 for r in results if r.chunk_id}
    out: dict[str, float] = {}
    for r in results:
        val = getattr(r, key)
        if isinstance(val, (int, float)):
            out[r.chunk_id] = (val - min_s) / (max_s - min_s)
    return out


def _apply_mmr(embedder: OpenAIEmbeddings, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
    if not results:
        return []
    texts = [r.doc.page_content for r in results]
    doc_embeddings = np.asarray(embedder.embed_documents(texts))
    query_embedding = np.asarray(embedder.embed_query(query))
    if doc_embeddings.size == 0 or query_embedding.size == 0:
        return results[:k]
    mmr_indices = maximal_marginal_relevance(
        query_embedding,
        doc_embeddings,
        lambda_mult=0.5,
        k=min(k, len(doc_embeddings)),
    )
    return [results[i] for i in mmr_indices]


class DenseRetriever:
    def __init__(self, chroma: Chroma, search_type: str):
        self.chroma = chroma
        self.search_type = search_type

    def retrieve(self, query: str, k: int, fetch_k: int) -> List[RetrievalResult]:
        if self.search_type == "mmr":
            retriever = self.chroma.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})
            docs = retriever.invoke(query)
            return [RetrievalResult(doc=d, score=None, chunk_id=metadata_to_chunk_id(d.metadata)) for d in docs]

        docs_scores = self.chroma.similarity_search_with_score(query, k=k)
        return [
            RetrievalResult(doc=doc, score=float(score) if score is not None else None, chunk_id=metadata_to_chunk_id(doc.metadata))
            for doc, score in docs_scores
        ]


class BM25Retriever:
    def __init__(self, bm25_index: BM25Index, chunk_map: dict[str, Document], search_type: str, embedder: OpenAIEmbeddings):
        self.bm25_index = bm25_index
        self.chunk_map = chunk_map
        self.search_type = search_type
        self.embedder = embedder

    def retrieve(self, query: str, k: int, fetch_k: int) -> List[RetrievalResult]:
        results = self.bm25_index.search(query, top_k=fetch_k)
        scored: list[RetrievalResult] = []
        for r in results:
            cid = r.get("chunk_id")
            doc = self.chunk_map.get(str(cid))
            if doc is None:
                continue
            scored.append(
                RetrievalResult(
                    doc=doc,
                    score=float(r.get("score")) if r.get("score") is not None else None,
                    chunk_id=str(cid),
                    score_sparse=float(r.get("score")) if r.get("score") is not None else None,
                )
            )

        if self.search_type == "mmr":
            return _apply_mmr(self.embedder, query, scored, k)

        return scored[:k]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        embedder: OpenAIEmbeddings,
        alpha: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder
        self.alpha = alpha

    def retrieve(self, query: str, k: int, fetch_k: int) -> List[RetrievalResult]:
        dense_results = self.dense_retriever.retrieve(query, k=fetch_k, fetch_k=fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=fetch_k, fetch_k=fetch_k)

        dense_norm = _normalize_scores(dense_results, "score")
        sparse_norm = _normalize_scores(bm25_results, "score_sparse")

        combined_map: dict[str, RetrievalResult] = {}
        for res in dense_results:
            if not res.chunk_id:
                continue
            combined_map[res.chunk_id] = res
            res.score_dense = res.score
        for res in bm25_results:
            if not res.chunk_id:
                continue
            existing = combined_map.get(res.chunk_id)
            if existing:
                existing.score_sparse = res.score
            else:
                combined_map[res.chunk_id] = res

        combined: list[RetrievalResult] = []
        for cid, res in combined_map.items():
            d_norm = dense_norm.get(cid)
            s_norm = sparse_norm.get(cid)
            if d_norm is None and s_norm is None:
                continue
            if d_norm is None:
                hybrid_score = s_norm
            elif s_norm is None:
                hybrid_score = d_norm
            else:
                hybrid_score = self.alpha * d_norm + (1 - self.alpha) * s_norm
            res.score_hybrid = hybrid_score
            combined.append(res)

        combined_sorted = sorted(combined, key=lambda r: r.score_hybrid or 0.0, reverse=True)

        if self.dense_retriever.search_type == "mmr":
            return _apply_mmr(self.embedder, query, combined_sorted, k)
        return combined_sorted[:k]


def build_retriever(
    settings: Settings,
    collection_name: str,
    retrieval_mode: str,
    search_type: str,
    k: int,
    fetch_k: int,
    alpha: float = 0.5,
) -> RetrieverInterface:
    if retrieval_mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    if search_type not in {"similarity", "mmr"}:
        raise ValueError(f"Unsupported search_type: {search_type}")

    embedder = OpenAIEmbeddings(model=settings.embedding_model)
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(settings.chroma_dir),
    )

    if retrieval_mode == "dense":
        return DenseRetriever(chroma=chroma, search_type=search_type)

    # Build BM25 index from chunks
    index_run_id = os.getenv("INDEX_RUN_ID")
    chunks_dir = settings.artifacts_dir / "chunks"
    if index_run_id:
        chunks_glob = str(chunks_dir / f"{index_run_id}.jsonl")
    else:
        chunks_glob = str(chunks_dir / "*.jsonl")
    bm25_dir = settings.artifacts_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    index_path = bm25_dir / (f"{collection_name}_{index_run_id}.pkl" if index_run_id else f"{collection_name}.pkl")

    bm25_index = build_or_load_bm25_index(index_path=str(index_path), chunks_glob=chunks_glob)
    chunks = _load_chunks(chunks_glob)
    chunk_map = _load_chunk_documents(chunks)
    bm25_retriever = BM25Retriever(bm25_index=bm25_index, chunk_map=chunk_map, search_type=search_type, embedder=embedder)

    if retrieval_mode == "bm25":
        return bm25_retriever

    dense_retriever = DenseRetriever(chroma=chroma, search_type=search_type)
    return HybridRetriever(dense_retriever=dense_retriever, bm25_retriever=bm25_retriever, embedder=embedder, alpha=alpha)
