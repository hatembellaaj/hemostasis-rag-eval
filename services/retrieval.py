from __future__ import annotations

import glob
import json
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.bm25_index import BM25Index


SearchType = Literal["similarity", "mmr"]


def build_filter(country: str | None, disease: str | None, year: str | None, study_type: str | None) -> Dict[str, Any] | None:
    f: Dict[str, Any] = {}
    if country:
        f["country"] = country
    if disease:
        f["disease"] = disease
    if year:
        f["year"] = year
    if study_type:
        f["study_type"] = study_type
    return f or None


def retrieve(
    vs: Chroma,
    query: str,
    search_type: SearchType,
    k: int,
    fetch_k: int,
    where: Dict[str, Any] | None = None,
) -> List[Document]:
    if search_type == "mmr":
        # Chroma supports max_marginal_relevance_search in LC wrapper
        return vs.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)
    else:
        return vs.similarity_search(query, k=k, filter=where)


def metadata_to_chunk_id(md: Dict[str, Any]) -> Optional[str]:
    if not md:
        return None
    if md.get("chunk_id"):
        return str(md["chunk_id"])
    doc_id = md.get("doc_id")
    chunk_idx = md.get("chunk_idx")
    if doc_id is not None and chunk_idx is not None:
        return f"{doc_id}::chunk::{chunk_idx}"
    return None


def load_chunk_documents(
    chunks_glob: str = "artifacts/chunks/*.jsonl", text_field: str = "text"
) -> Dict[str, Document]:
    docs: Dict[str, Document] = {}
    for path_str in sorted(glob.glob(chunks_glob)):
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                cid = chunk.get("chunk_id")
                if not cid:
                    continue
                text = chunk.get(text_field) or chunk.get("text") or chunk.get("text_block")
                if not text:
                    continue
                md = chunk.get("metadata") or {}
                md = dict(md)
                md["chunk_id"] = cid
                docs[str(cid)] = Document(page_content=str(text), metadata=md)
    return docs


class DenseRetriever:
    def __init__(self, vs: Chroma, search_type: SearchType = "similarity"):
        self.vs = vs
        self.search_type = search_type

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.search_type == "mmr":
            docs = self.vs.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 2, k))
            results = []
            for doc in docs:
                results.append(
                    {
                        "chunk_id": metadata_to_chunk_id(doc.metadata) or "",
                        "score": None,
                        "doc": doc,
                    }
                )
            return results

        docs_scores = self.vs.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_scores:
            results.append(
                {
                    "chunk_id": metadata_to_chunk_id(doc.metadata) or "",
                    "score": float(score) if score is not None else None,
                    "doc": doc,
                }
            )
        return results


class SparseBM25Retriever:
    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 20):
        results = self.bm25_index.search(query, top_k=top_k)
        for r in results:
            r["score_sparse"] = r.pop("score")
        return results


def _normalize_scores(results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    scores = [r.get(key) for r in results if isinstance(r.get(key), (int, float)) and isfinite(r.get(key))]
    if not scores:
        return {}
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return {r["chunk_id"]: 1.0 for r in results if r.get(key) is not None}
    out: Dict[str, float] = {}
    for r in results:
        val = r.get(key)
        if isinstance(val, (int, float)) and isfinite(val):
            out[r["chunk_id"]] = (val - min_s) / (max_s - min_s)
    return out


def hybrid_retrieve(
    query: str,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseBM25Retriever,
    k_dense: int = 50,
    k_sparse: int = 50,
    k_final: int = 20,
    alpha: float = 0.5,
):
    dense_results = dense_retriever.retrieve(query, k=k_dense)
    sparse_results = sparse_retriever.retrieve(query, top_k=k_sparse)

    dense_norm = _normalize_scores(dense_results, "score")
    sparse_norm = _normalize_scores(sparse_results, "score_sparse")

    dense_map = {r.get("chunk_id"): r for r in dense_results if r.get("chunk_id")}
    sparse_map = {r.get("chunk_id"): r for r in sparse_results if r.get("chunk_id")}

    combined = []
    for cid in set(dense_map) | set(sparse_map):
        d_norm = dense_norm.get(cid)
        s_norm = sparse_norm.get(cid)

        if d_norm is None and s_norm is None:
            continue

        score_dense = dense_map.get(cid, {}).get("score")
        score_sparse = sparse_map.get(cid, {}).get("score_sparse")

        if d_norm is None:
            hybrid_score = s_norm
        elif s_norm is None:
            hybrid_score = d_norm
        else:
            hybrid_score = alpha * d_norm + (1 - alpha) * s_norm

        combined.append(
            {
                "chunk_id": cid,
                "score_dense": score_dense,
                "score_sparse": score_sparse,
                "score_hybrid": hybrid_score,
                "doc": dense_map.get(cid, {}).get("doc"),
            }
        )

    combined_sorted = sorted(combined, key=lambda r: r.get("score_hybrid", 0), reverse=True)
    return combined_sorted[:k_final]
