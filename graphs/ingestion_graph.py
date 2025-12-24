from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from services.loaders import load_paper, load_registry_excel
from services.registry_to_text import registry_views_to_pivot_rows
from services.storage import Settings, ensure_dirs, new_id, now_iso, write_json, write_jsonl
from services.chunking import clean_text


class IngestionState(TypedDict, total=False):
    # inputs
    settings: Settings
    paper_paths: List[str]          # list of file paths
    registry_excel_path: Optional[str]

    # intermediate
    paper_docs: List[Document]      # loaded docs (pages)
    registry_rows: List[Dict[str, Any]]

    # outputs
    snapshot_id: str
    pivot_rows: List[Dict[str, Any]]
    pivot_path: str
    stats: Dict[str, Any]


def _node_init(state: IngestionState) -> IngestionState:
    settings = state["settings"]
    ensure_dirs(settings)
    snapshot_id = new_id("snapshot")
    return {
        **state,
        "snapshot_id": snapshot_id,
        "stats": {
            "created_at": now_iso(),
            "snapshot_id": snapshot_id,
            "papers_n": 0,
            "paper_pages_n": 0,
            "registry_rows_n": 0,
        },
    }


def _node_load_papers(state: IngestionState) -> IngestionState:
    paper_paths = [Path(p) for p in (state.get("paper_paths") or [])]
    docs: List[Document] = []
    for p in paper_paths:
        if not p.exists():
            continue
        res = load_paper(p)
        docs.extend(res.docs)

    st = dict(state["stats"])
    st["papers_n"] = len(paper_paths)
    st["paper_pages_n"] = len(docs)

    return {**state, "paper_docs": docs, "stats": st}


def _node_load_registry(state: IngestionState) -> IngestionState:
    registry_excel_path = state.get("registry_excel_path")
    if not registry_excel_path:
        return {**state, "registry_rows": []}

    p = Path(registry_excel_path)
    if not p.exists():
        return {**state, "registry_rows": []}

    reg = load_registry_excel(p)
    rows = registry_views_to_pivot_rows(reg.raw_tables, reg.source_file)

    st = dict(state["stats"])
    st["registry_rows_n"] = len(rows)

    return {**state, "registry_rows": rows, "stats": st}


def _node_build_pivot(state: IngestionState) -> IngestionState:
    snapshot_id = state["snapshot_id"]

    rows: List[Dict[str, Any]] = []

    # 1) papers -> pivot rows (page-level)
    for i, d in enumerate(state.get("paper_docs") or []):
        text = clean_text(d.page_content or "")
        md = dict(d.metadata or {})
        source_file = md.get("source_file")

        row = {
            "doc_id": f"paper::{source_file}::page::{md.get('page', i)}::{i}",
            "source_type": "paper",
            "source": source_file,
            "text": text,
            "provenance": {
                "page": md.get("page", None),
                "source_file": source_file,
            },
            "metadata": {
                # NOTE: single-country setting: registry is local (Tunisia),
                # literature is global -> country None by default
                "country": None,
                "disease": None,       # can be filled later if you tag papers
                "time_window": None,
                "study_type": "paper",
            },
            "snapshot_id": snapshot_id,
        }
        rows.append(row)

    # 2) registry rows already pivot-compatible
    for r in state.get("registry_rows") or []:
        r2 = dict(r)
        r2["snapshot_id"] = snapshot_id
        rows.append(r2)

    return {**state, "pivot_rows": rows}


def _node_persist_pivot(state: IngestionState) -> IngestionState:
    settings = state["settings"]
    snapshot_id = state["snapshot_id"]
    pivot_rows = state.get("pivot_rows") or []

    pivot_path = settings.artifacts_dir / "pivot" / f"{snapshot_id}.jsonl"
    write_jsonl(pivot_path, pivot_rows)

    # write stats sidecar
    stats = dict(state.get("stats") or {})
    stats["pivot_rows_n"] = len(pivot_rows)
    stats_path = settings.artifacts_dir / "stats" / f"{snapshot_id}.json"
    write_json(stats_path, stats)

    return {**state, "pivot_path": str(pivot_path)}


def build_ingestion_graph() -> Any:
    g = StateGraph(IngestionState)

    g.add_node("init", _node_init)
    g.add_node("load_papers", _node_load_papers)
    g.add_node("load_registry", _node_load_registry)
    g.add_node("build_pivot", _node_build_pivot)
    g.add_node("persist_pivot", _node_persist_pivot)

    g.set_entry_point("init")
    g.add_edge("init", "load_papers")
    g.add_edge("load_papers", "load_registry")
    g.add_edge("load_registry", "build_pivot")
    g.add_edge("build_pivot", "persist_pivot")
    g.add_edge("persist_pivot", END)

    return g.compile()
