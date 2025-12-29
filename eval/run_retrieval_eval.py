from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from eval.metrics import aggregate_metrics, compute_metrics_for_results
from eval.plots import generate_figures
from eval.retrievers import RetrievalResult, build_retriever
from services.storage import Settings, new_id, now_iso, write_json, write_jsonl


def parse_grid(grid: str, default_alpha: float) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    for entry in grid.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 2:
            continue
        retrieval_mode, search_type = parts[0], parts[1]
        alpha = float(parts[2]) if len(parts) >= 3 else default_alpha
        configs.append({"retrieval_mode": retrieval_mode, "search_type": search_type, "alpha": alpha})
    return configs


def load_questions(path: Path, max_questions: Optional[int] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_questions and len(rows) >= max_questions:
                break
    return rows


def build_retrieved_topk(results: List[RetrievalResult]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx, res in enumerate(results, start=1):
        md = dict(res.doc.metadata or {})
        preview = (res.doc.page_content or "")[:300]
        out.append(
            {
                "rank": idx,
                "score": res.score_hybrid or res.score_dense or res.score_sparse or res.score,
                "chunk_id": res.chunk_id,
                "doc_id": md.get("doc_id"),
                "source": md.get("source"),
                "metadata": md,
                "text_preview": preview,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation grid")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--collection", type=str, default="docs")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch_k", type=int, default=None)
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--index_run_id", type=str, default=None)
    parser.add_argument("--snapshot_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--default_alpha", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)

    settings = Settings.load()
    out_dir = args.out or settings.artifacts_dir / "eval_retrieval"
    out_dir = Path(out_dir)

    eval_id = new_id("retrieval_eval")

    if args.index_run_id:
        os.environ["INDEX_RUN_ID"] = args.index_run_id

    questions = load_questions(args.questions, max_questions=args.max_questions)

    configs = parse_grid(args.grid, args.default_alpha)
    if not configs:
        raise ValueError("No valid configs parsed from grid")

    runs_dir = out_dir / "runs"
    summaries_dir = out_dir / "summaries"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []

    config_index: Dict[str, Dict[str, object]] = {}

    for config in configs:
        retrieval_mode = config["retrieval_mode"]
        search_type = config["search_type"]
        alpha = float(config.get("alpha", args.default_alpha))

        effective_fetch_k = args.fetch_k
        if effective_fetch_k is None:
            effective_fetch_k = 12 if search_type == "similarity" else 20

        retriever = build_retriever(
            settings=settings,
            collection_name=args.collection,
            retrieval_mode=retrieval_mode,
            search_type=search_type,
            k=args.k,
            fetch_k=effective_fetch_k,
            alpha=alpha,
        )

        config_id = f"{retrieval_mode}_{search_type}_k{args.k}_f{effective_fetch_k}_a{alpha}"
        config_index[config_id] = {
            "retrieval_mode": retrieval_mode,
            "search_type": search_type,
            "alpha": alpha,
            "k": args.k,
            "fetch_k": effective_fetch_k,
        }

        for q in questions:
            results = retriever.retrieve(q.get("question", ""), k=args.k, fetch_k=effective_fetch_k)
            metrics = compute_metrics_for_results(results, q)
            row = {
                "eval_id": eval_id,
                "config_id": config_id,
                "retrieval_mode": retrieval_mode,
                "search_type": search_type,
                "alpha": alpha,
                "k": args.k,
                "fetch_k": effective_fetch_k,
                "index_run_id": args.index_run_id,
                "snapshot_id": args.snapshot_id,
                "question_id": q.get("id"),
                "use_case": q.get("use_case"),
                "hit_at_k": metrics.hit_at_k,
                "recall_at_k": metrics.recall_at_k,
                "files_hit_at_k": metrics.files_hit_at_k,
                "duplication_at_k_exact": metrics.duplication_at_k_exact,
                "unique_doc_rate_at_k": metrics.unique_doc_rate_at_k,
                "context_chars_at_k": metrics.context_chars_at_k,
                "avg_chunk_len_at_k": metrics.avg_chunk_len_at_k,
                "hints_covered": metrics.hints_covered,
                "total_hints": metrics.total_hints,
                "retrieved_topk": build_retrieved_topk(results),
            }
            run_rows.append(row)

    runs_path = runs_dir / f"{eval_id}.jsonl"
    write_jsonl(runs_path, run_rows)

    summary_rows: List[Dict[str, object]] = []
    config_summaries = aggregate_metrics(run_rows, "config_id")
    for cfg_summary in config_summaries:
        cfg_info = config_index.get(cfg_summary["config_id"], {})
        cfg_summary.update(
            {
                "scope": "overall",
                "retrieval_mode": cfg_info.get("retrieval_mode"),
                "search_type": cfg_info.get("search_type"),
                "alpha": cfg_info.get("alpha"),
                "k": cfg_info.get("k", args.k),
                "fetch_k": cfg_info.get("fetch_k", args.fetch_k),
            }
        )
        summary_rows.append(cfg_summary)

    # By use case
    for config_id, cfg in config_index.items():
        per_config = [r for r in run_rows if r["config_id"] == config_id]
        uc_summary = aggregate_metrics(per_config, "use_case")
        for uc in uc_summary:
            uc.update(
                {
                    "scope": "use_case",
                    "retrieval_mode": cfg["retrieval_mode"],
                    "search_type": cfg["search_type"],
                    "alpha": cfg.get("alpha", args.default_alpha),
                    "k": args.k,
                    "fetch_k": cfg["fetch_k"],
                    "config_id": config_id,
                    "use_case": uc.get("use_case"),
                }
            )
            summary_rows.append(uc)

    summary_csv_path = tables_dir / f"{eval_id}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    summary_json = {
        "eval_id": eval_id,
        "created_at": now_iso(),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "collection_name": args.collection,
        "snapshot_id": args.snapshot_id,
        "index_run_id": args.index_run_id,
        "k": args.k,
        "fetch_k": args.fetch_k,
        "eval_grid": configs,
        "runs_path": str(runs_path),
        "tables_path": str(summary_csv_path),
        "text_block": f"All retrieval comparisons were run on I3 index_run_id={args.index_run_id}; embedding_model={settings.embedding_model}; k={args.k}; fetch_k={args.fetch_k or 'auto' }.",
        "use_case_note": "UC breakdown available in CSV/figures",
        "config_summaries": config_summaries,
    }
    summary_json_path = summaries_dir / f"{eval_id}.json"
    write_json(summary_json_path, summary_json)

    generate_figures(summary_csv_path, figures_dir)

    print(f"Evaluation completed. Eval ID: {eval_id}")
    print(f"Runs written to: {runs_path}")
    print(f"Summary table: {summary_csv_path}")
    print(f"Summary json: {summary_json_path}")


if __name__ == "__main__":
    main()
