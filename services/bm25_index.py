from __future__ import annotations

import glob
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    def __init__(self, chunk_ids: List[str], corpus_tokens: List[List[str]]):
        self.chunk_ids = chunk_ids
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens)

    @classmethod
    def from_chunks(
        cls, chunks: List[Dict[str, Any]], text_field: str = "text"
    ) -> "BM25Index":
        chunk_ids: List[str] = []
        corpus_tokens: List[List[str]] = []

        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            text = chunk.get(text_field) or chunk.get("text") or chunk.get("text_block")
            if not text:
                continue
            chunk_ids.append(str(cid))
            corpus_tokens.append(tokenize(str(text)))

        if not chunk_ids:
            raise ValueError("No valid chunks provided to build BM25 index")

        return cls(chunk_ids=chunk_ids, corpus_tokens=corpus_tokens)

    def save(self, path: str) -> None:
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(chunk_ids=data["chunk_ids"], corpus_tokens=data["corpus_tokens"])
        idx.bm25 = data["bm25"]
        return idx

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for rank, (cid, score) in enumerate(ranked[:top_k], start=1):
            out.append({"chunk_id": cid, "score": float(score), "rank": rank})
        return out


def build_or_load_bm25_index(
    index_path: str = "artifacts/bm25_index.pkl",
    chunks_glob: str = "artifacts/chunks/*.jsonl",
    text_field: str = "text",
) -> BM25Index:
    idx_path = Path(index_path)
    if idx_path.exists():
        return BM25Index.load(str(idx_path))

    chunk_files = [Path(p) for p in glob.glob(chunks_glob)]
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found matching glob: {chunks_glob}")

    chunks: List[Dict[str, Any]] = []
    for path in chunk_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunks.append(chunk)

    idx = BM25Index.from_chunks(chunks=chunks, text_field=text_field)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(idx_path))
    return idx
