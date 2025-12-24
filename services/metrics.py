from __future__ import annotations

import hashlib
from statistics import mean
from typing import Any, Dict, List

from langchain_core.documents import Document


def chunk_stats(chunks: List[Document]) -> Dict[str, Any]:
    if not chunks:
        return {"n_chunks": 0}

    lengths = [len(c.page_content or "") for c in chunks]
    empty_rate = sum(1 for L in lengths if L < 50) / max(1, len(lengths))

    # duplication: exact hash duplicates (cheap proxy)
    hashes = [hashlib.md5((c.page_content or "").encode("utf-8")).hexdigest() for c in chunks]
    dup_rate = 1.0 - (len(set(hashes)) / max(1, len(hashes)))

    # metadata completeness proxies
    def has_country(c: Document) -> bool:
        md = c.metadata or {}
        return bool(md.get("country") or md.get("metadata", {}).get("country"))

    country_rate = sum(1 for c in chunks if has_country(c)) / max(1, len(chunks))

    return {
        "n_chunks": len(chunks),
        "len_min": min(lengths),
        "len_mean": int(mean(lengths)),
        "len_max": max(lengths),
        "empty_rate_lt_50": round(empty_rate, 4),
        "dup_rate_exact": round(dup_rate, 4),
        "country_tag_rate": round(country_rate, 4),
    }


def context_cost(context: str) -> Dict[str, Any]:
    return {
        "context_chars": len(context),
        "context_words": len(context.split()),
    }
