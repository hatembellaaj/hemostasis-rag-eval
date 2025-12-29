from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from eval.retrievers import RetrievalResult


HINT_NORMALIZER = re.compile(r"\s+")


@dataclass
class RetrievalMetrics:
    hit_at_k: int
    recall_at_k: float
    files_hit_at_k: int
    duplication_at_k_exact: float
    unique_doc_rate_at_k: float
    context_chars_at_k: int
    avg_chunk_len_at_k: float
    hints_covered: int
    total_hints: int


def _normalize_text(text: str) -> str:
    return HINT_NORMALIZER.sub(" ", text.strip().lower())


def _hash_text(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def _match_hints(text: str, hints: Iterable[str]) -> set[str]:
    normalized_text = _normalize_text(text)
    matched: set[str] = set()
    for hint in hints:
        if not hint:
            continue
        if _normalize_text(hint) in normalized_text:
            matched.add(hint)
    return matched


def _match_files(metadata: Dict[str, str], gold_files: Iterable[str]) -> bool:
    src = str(metadata.get("source") or metadata.get("doc_id") or "").lower()
    for gf in gold_files:
        if gf and str(gf).lower() in src:
            return True
    return False


def compute_metrics_for_results(
    results: List[RetrievalResult],
    question: Dict[str, object],
) -> RetrievalMetrics:
    gold_hints: list[str] = [str(h) for h in (question.get("gold_hints") or [])]
    gold_files: list[str] = [str(f) for f in (question.get("gold_files") or [])]

    hints_found: set[str] = set()
    file_hit = False

    chunk_hashes: Counter[str] = Counter()
    doc_ids: set[str] = set()

    context_chars = 0

    for res in results:
        text = res.doc.page_content or ""
        context_chars += len(text)
        hints_found.update(_match_hints(text, gold_hints))
        file_hit = file_hit or _match_files(res.doc.metadata or {}, gold_files)

        chunk_hashes[_hash_text(text)] += 1
        doc_id = str(res.doc.metadata.get("doc_id") or res.doc.metadata.get("source") or "")
        if doc_id:
            doc_ids.add(doc_id)

    total_hints = len(gold_hints)
    recall_at_k = (len(hints_found) / total_hints) if total_hints else 0.0
    hit_at_k = 1 if hints_found or file_hit else 0
    files_hit_at_k = 1 if file_hit else 0

    duplicates = sum(count - 1 for count in chunk_hashes.values())
    denom = max(len(results), 1)
    duplication_at_k_exact = duplicates / denom
    unique_doc_rate_at_k = (len(doc_ids) / denom) if denom else 0.0
    avg_chunk_len_at_k = (context_chars / denom) if denom else 0.0

    return RetrievalMetrics(
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        files_hit_at_k=files_hit_at_k,
        duplication_at_k_exact=duplication_at_k_exact,
        unique_doc_rate_at_k=unique_doc_rate_at_k,
        context_chars_at_k=context_chars,
        avg_chunk_len_at_k=avg_chunk_len_at_k,
        hints_covered=len(hints_found),
        total_hints=total_hints,
    )


def aggregate_metrics(rows: List[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    summaries: list[Dict[str, object]] = []
    for key, items in grouped.items():
        n = len(items)
        if n == 0:
            continue
        summary: Dict[str, object] = {
            group_key: key,
            "n_questions": n,
            "hit_at_k_rate": sum(int(i.get("hit_at_k", 0)) for i in items) / n,
            "recall_at_k_mean": sum(float(i.get("recall_at_k", 0)) for i in items) / n,
            "files_hit_at_k_rate": sum(int(i.get("files_hit_at_k", 0)) for i in items) / n,
            "duplication_at_k_mean": sum(float(i.get("duplication_at_k_exact", 0)) for i in items) / n,
            "unique_doc_rate_mean": sum(float(i.get("unique_doc_rate_at_k", 0)) for i in items) / n,
            "context_chars_mean": sum(int(i.get("context_chars_at_k", 0)) for i in items) / n,
            "avg_chunk_len_mean": sum(float(i.get("avg_chunk_len_at_k", 0)) for i in items) / n,
        }
        summaries.append(summary)
    return summaries
