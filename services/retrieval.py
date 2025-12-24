from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document


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
