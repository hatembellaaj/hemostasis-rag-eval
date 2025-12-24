from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _get(df: pd.DataFrame, col: str, default=None):
    return df[col] if col in df.columns else default


def registry_views_to_pivot_rows(
    tables: Dict[str, pd.DataFrame],
    source_file: str,
) -> List[Dict[str, Any]]:
    """
    Convert registry Excel sheets into pivot JSONL rows.
    Recommended: a sheet named 'views' with columns:
      country, disease, time_window (or year), metric_name, metric_value,
      unit(optional), cohort(optional), n(optional), notes(optional)
    """
    if "views" not in tables:
        raise ValueError(
            "Excel must contain a sheet named 'views' (recommended minimal input)."
        )

    df = tables["views"].copy()

    required = ["country", "disease", "metric_name", "metric_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in 'views': {missing}")

    # Normalize column variants
    if "time_window" not in df.columns and "year" in df.columns:
        df["time_window"] = df["year"].astype(str)

    rows: List[Dict[str, Any]] = []
    for i, r in df.iterrows():
        country = str(r.get("country", "")).strip()
        disease = str(r.get("disease", "")).strip()
        time_window = str(r.get("time_window", "")).strip()
        metric_name = str(r.get("metric_name", "")).strip()
        metric_value = r.get("metric_value", None)
        unit = str(r.get("unit", "")).strip() if "unit" in df.columns else ""
        cohort = str(r.get("cohort", "")).strip() if "cohort" in df.columns else ""
        n = r.get("n", None) if "n" in df.columns else None
        notes = str(r.get("notes", "")).strip() if "notes" in df.columns else ""

        # Template -> text
        parts = []
        if cohort:
            parts.append(f"In cohort '{cohort}'")
        else:
            parts.append("In the registry cohort")

        if country:
            parts.append(f"in {country}")
        if time_window:
            parts.append(f"({time_window})")

        lead = " ".join(parts).strip()

        value_str = str(metric_value)
        if unit:
            value_str = f"{value_str} {unit}".strip()

        text = f"{lead}, {metric_name}: {value_str}."
        if notes:
            text += f" Notes: {notes}"

        row = {
            "doc_id": f"registry::{source_file}::views::{i}",
            "source_type": "registry_view",
            "source": source_file,
            "text": text,
            "provenance": {
                "sheet": "views",
                "row_index": int(i),
            },
            "metadata": {
                "country": country or None,
                "disease": disease or None,
                "time_window": time_window or None,
                "study_type": "registry",
                "cohort": cohort or None,
            },
            "numeric_payload": {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "unit": unit or None,
                "n": n,
            },
        }
        rows.append(row)

    return rows
