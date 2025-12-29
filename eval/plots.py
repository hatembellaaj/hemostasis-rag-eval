from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


PLOT_FIELDS = {
    "hit_at_k_rate": "Hit@k rate",
    "recall_at_k_mean": "Recall@k mean",
    "duplication_at_k_mean": "Duplication@k mean",
    "context_chars_mean": "Context chars@k",
}


def _ensure_figures_dir(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


def _plot_bar(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    plt.clf()
    ax = df.plot(kind="bar", x="config_id", y=metric, legend=False, figsize=(8, 4))
    ax.set_ylabel(title)
    ax.set_xlabel("Config")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)


def _plot_by_use_case(summary_df: pd.DataFrame, metric: str, title: str, figures_dir: Path) -> None:
    use_cases = sorted(summary_df["use_case"].dropna().unique())
    for uc in use_cases:
        sub = summary_df[(summary_df["use_case"] == uc) & (summary_df["scope"] == "use_case")]
        if sub.empty:
            continue
        out = figures_dir / f"{metric}_by_config_{uc}.png"
        _plot_bar(sub, metric, f"{title} - {uc}", out)


def generate_figures(summary_csv: Path, figures_dir: Path) -> List[Path]:
    summary_df = pd.read_csv(summary_csv)
    _ensure_figures_dir(figures_dir)

    created: List[Path] = []
    overall = summary_df[summary_df["scope"] == "overall"]

    for field, title in PLOT_FIELDS.items():
        out_path = figures_dir / f"{field}_by_config.png"
        _plot_bar(overall, field, title, out_path)
        created.append(out_path)
        _plot_by_use_case(summary_df, field, title, figures_dir)

    return created
