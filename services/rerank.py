from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document


def similarity_with_scores(vs: Chroma, query: str, k: int, where: Dict[str, Any] | None = None) -> List[Tuple[Document, float]]:
    return vs.similarity_search_with_score(query, k=k, filter=where)


def apply_threshold(
    docs_with_scores: List[Tuple[Document, float]],
    score_threshold: float | None,
) -> List[Tuple[Document, float]]:
    if score_threshold is None:
        return docs_with_scores

    # NOTE: Chroma score semantics depend on distance function.
    # We keep this as a tunable knob and document that it's empirical.
    return [(d, s) for (d, s) in docs_with_scores if s <= score_threshold]
