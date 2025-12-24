#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_registry_views.py

Convertit un Excel patient-level (registre) -> un Excel "views" (agrégats) prêt pour l'ingestion RAG.
- Supprime/ignore les colonnes identifiantes (nom, prénom, numéro dossier, date naissance, ville...)
- Calcule des métriques utiles protocole (hémophilie) : n, âge, sexe, sévérité, inhibiteur, prophylaxie, accidents, complications virales...
- Produit un fichier Excel avec une feuille 'views' attendue par la moulinette.

Usage:
  python make_registry_views.py \
      --input data/registry/registry_raw.xlsx \
      --output data/registry/registry_for_pipeline.xlsx \
      --country Tunisia \
      --time-window "2018-2024" \
      --include-sanitized

Dépendances:
  pandas, openpyxl (déjà dans requirements.txt du projet)
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Helpers: normalization & column matching
# -------------------------
def norm(s: str) -> str:
    """Lowercase, remove accents, trim, collapse spaces."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """
    Find a column in df that matches one of candidates after normalization.
    Example: find_col(df, "sexe", "sex")
    """
    cols = list(df.columns)
    norm_map = {norm(c): c for c in cols}
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def boolify(x: Any) -> Optional[bool]:
    """Convert common FR yes/no variants to bool."""
    if pd.isna(x):
        return None
    s = norm(str(x))
    if s in {"oui", "o", "yes", "y", "true", "vrai", "1"}:
        return True
    if s in {"non", "n", "no", "false", "faux", "0"}:
        return False
    # sometimes values are like "oui/non" or blanks
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric conversion (handles commas)."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# -------------------------
# Metric building
# -------------------------
def add_view(
    views: List[Dict[str, Any]],
    *,
    country: str,
    disease: str,
    time_window: str,
    cohort: str,
    metric_name: str,
    metric_value: Any,
    unit: Optional[str] = None,
    n: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    views.append(
        {
            "country": country,
            "disease": disease,
            "time_window": time_window,
            "cohort": cohort,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "unit": unit or "",
            "n": n if n is not None else "",
            "notes": notes or "",
        }
    )


def safe_value(x: Any) -> Any:
    """Avoid NaN in outputs."""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return x


def infer_disease_from_row(row: pd.Series) -> str:
    """
    Ici ton fichier ressemble fortement à l'hémophilie.
    On renvoie par défaut 'Hemophilia' ou 'Hemophilia A/B' si on peut.
    """
    # "forme d'hémophilie" peut contenir A/B
    for key in ["forme d'hémophilie", "forme d'hemophilie", "forme hemophilie"]:
        # not used directly; handled by column mapping outside
        pass
    return "Hemophilia"


def compute_views(df: pd.DataFrame, country: str, time_window: str) -> pd.DataFrame:
    """
    Calcule des vues agrégées sur le dataset hémophilie.
    On produit quelques métriques "protocole-oriented".
    """
    views: List[Dict[str, Any]] = []

    # --- Identify important columns (best-effort) ---
    col_age = find_col(df, "âge actuel", "age actuel", "age")
    col_sex = find_col(df, "sexe", "sex")
    col_severity = find_col(df, "sévérité", "severite", "severity")
    col_hemo_form = find_col(df, "forme d'hémophilie", "forme d'hemophilie", "forme hemophilie")
    col_factor8 = find_col(df, "facteur viii", "facteur 8", "fviii")
    col_factor9 = find_col(df, "facteur ix", "facteur 9", "fix")
    col_inhib = find_col(df, "inhibiteur", "inhibitor")
    col_inhib_titer = find_col(df, "taux d'inhibiteur", "taux inhibiteur", "inhibitor titer")
    col_prophy_or_od = find_col(
        df,
        "modalité du ttt substitutif prophylactique ou a la demande",
        "modalite du ttt substitutif prophylactique ou a la demande",
        "prophylactique ou a la demande",
        "prophylaxie ou a la demande",
    )
    col_hemlibra = find_col(df, "prophylaxie par hemlibra", "hemlibra")
    col_bleeds6m = find_col(df, "nbre d'accidents hémorragique/6 mois", "nbre d accidents hemorragique/6 mois")
    col_hbv = find_col(df, "sérologie hbv", "serologie hbv", "hbv")
    col_hcv = find_col(df, "sérologie vhc", "serologie vhc", "vhc", "hcv")
    col_hiv = find_col(df, "sérologie hiv", "serologie hiv", "hiv")

    # Helper to define cohorts by hemophilia type if available
    if col_hemo_form:
        df["_hemo_form"] = df[col_hemo_form].astype(str).apply(norm)
    else:
        df["_hemo_form"] = ""

    # Basic disease labels:
    # If "A" or "VIII" in form -> Hemophilia A, if "B" or "IX" -> Hemophilia B, else Hemophilia
    def label_disease_from_form(form: str) -> str:
        if any(k in form for k in [" a", "type a", "viii", "8", "fviii"]):
            return "Hemophilia A"
        if any(k in form for k in [" b", "type b", "ix", "9", "fix"]):
            return "Hemophilia B"
        return "Hemophilia"

    df["_disease"] = df["_hemo_form"].apply(label_disease_from_form)

    # We will produce views overall + by disease subgroup (A/B/unknown)
    cohorts = [("all", df)]
    for dis in sorted([d for d in df["_disease"].unique() if d]):
        cohorts.append((dis.lower().replace(" ", "_"), df[df["_disease"] == dis]))

    # --- Compute metrics per cohort ---
    for cohort_name, sub in cohorts:
        n_pat = int(len(sub))
        if n_pat == 0:
            continue

        # Choose disease label
        if cohort_name == "all":
            disease = "Hemophilia"
        else:
            # cohort_name is based on disease; get actual
            disease = sub["_disease"].iloc[0] if "_disease" in sub.columns and len(sub) else "Hemophilia"

        add_view(
            views,
            country=country,
            disease=disease,
            time_window=time_window,
            cohort=cohort_name,
            metric_name="Number of registered patients",
            metric_value=n_pat,
            unit="count",
            n=n_pat,
        )

        # Age
        if col_age:
            age = to_numeric(sub[col_age])
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Age (mean)", metric_value=round(float(age.mean()), 2) if age.notna().any() else "",
                     unit="years", n=n_pat)
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Age (median)", metric_value=round(float(age.median()), 2) if age.notna().any() else "",
                     unit="years", n=n_pat)

        # Sex distribution
        if col_sex:
            sex = sub[col_sex].astype(str).apply(norm)
            male = int((sex == "m").sum() + (sex == "masculin").sum())
            female = int((sex == "f").sum() + (sex == "feminin").sum() + (sex == "féminin").sum())
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Sex: male count", metric_value=male, unit="count", n=n_pat)
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Sex: female count", metric_value=female, unit="count", n=n_pat)

        # Severity distribution
        if col_severity:
            sev = sub[col_severity].astype(str).apply(norm)
            # common: "severe", "moderee", "legere" etc.
            severe = int(sev.str.contains("sev").sum())
            moderate = int(sev.str.contains("mod").sum())
            mild = int(sev.str.contains("leg").sum() + sev.str.contains("mild").sum())
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Severity: severe count", metric_value=severe, unit="count", n=n_pat)
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Severity: moderate count", metric_value=moderate, unit="count", n=n_pat)
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Severity: mild count", metric_value=mild, unit="count", n=n_pat)

        # Factor levels (only if columns exist)
        if col_factor8 and (disease in {"Hemophilia", "Hemophilia A"}):
            f8 = to_numeric(sub[col_factor8])
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Factor VIII level (median)", metric_value=round(float(f8.median()), 4) if f8.notna().any() else "",
                     unit="IU/dL (as recorded)", n=n_pat)
        if col_factor9 and (disease in {"Hemophilia", "Hemophilia B"}):
            f9 = to_numeric(sub[col_factor9])
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Factor IX level (median)", metric_value=round(float(f9.median()), 4) if f9.notna().any() else "",
                     unit="IU/dL (as recorded)", n=n_pat)

        # Inhibitor prevalence
        if col_inhib:
            inhib = sub[col_inhib].apply(boolify)
            if inhib.notna().any():
                prev = float(inhib.mean())  # True=1, False=0
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name="Inhibitor prevalence", metric_value=round(prev, 4),
                         unit="proportion", n=n_pat)

        # Inhibitor titer summary
        if col_inhib_titer:
            tit = to_numeric(sub[col_inhib_titer])
            if tit.notna().any():
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name="Inhibitor titer (median)", metric_value=round(float(tit.median()), 4),
                         unit="BU (as recorded)", n=n_pat)

        # Prophylaxis vs on-demand
        if col_prophy_or_od:
            mode = sub[col_prophy_or_od].astype(str).apply(norm)
            prophy = int(mode.str.contains("proph").sum())
            ondemand = int(mode.str.contains("demande").sum() + mode.str.contains("a la demande").sum() + mode.str.contains("a la demande").sum())
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Treatment modality: prophylaxis count", metric_value=prophy,
                     unit="count", n=n_pat)
            add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                     metric_name="Treatment modality: on-demand count", metric_value=ondemand,
                     unit="count", n=n_pat)

        # Hemlibra usage
        if col_hemlibra:
            heml = sub[col_hemlibra].apply(boolify)
            if heml.notna().any():
                rate = float(heml.mean())
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name="Hemlibra prophylaxis rate", metric_value=round(rate, 4),
                         unit="proportion", n=n_pat)

        # Bleeding events in last 6 months
        if col_bleeds6m:
            b6 = to_numeric(sub[col_bleeds6m])
            if b6.notna().any():
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name="Bleeding events per 6 months (mean)", metric_value=round(float(b6.mean()), 3),
                         unit="count", n=n_pat)
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name="Bleeding events per 6 months (median)", metric_value=round(float(b6.median()), 3),
                         unit="count", n=n_pat)

        # Viral serology presence (very rough proxies)
        def sero_rate(col: Optional[str], label: str) -> None:
            if not col:
                return
            s = sub[col].astype(str).apply(norm)
            # classify positive if contains "pos" or "reactif"
            pos = s.str.contains("pos").sum() + s.str.contains("react").sum()
            tot = int(len(s))
            if tot > 0:
                add_view(views, country=country, disease=disease, time_window=time_window, cohort=cohort_name,
                         metric_name=f"{label} positive (proxy)", metric_value=round(float(pos / tot), 4),
                         unit="proportion", n=tot,
                         notes="Proxy based on string matching ('pos', 'reactif'). Verify mapping.")
        sero_rate(col_hbv, "HBV serology")
        sero_rate(col_hcv, "HCV serology")
        sero_rate(col_hiv, "HIV serology")

    out = pd.DataFrame(views)
    # Ensure required columns exist exactly as expected by our ingestion module
    # required: country, disease, metric_name, metric_value; time_window recommended
    return out


# -------------------------
# Sanitized patients export (optional)
# -------------------------
IDENTIFYING_PATTERNS = [
    r"\bnom\b",
    r"\bprenom\b",
    r"\bpr[eé]nom\b",
    r"num[eé]ro du dossier",
    r"\bdossier\b",
    r"date de naissance",
    r"\bville\b",
    r"\borigine\b",
]


def sanitized_patients(df: pd.DataFrame) -> pd.DataFrame:
    cols_keep = []
    for c in df.columns:
        nc = norm(c)
        if any(re.search(p, nc) for p in IDENTIFYING_PATTERNS):
            continue
        cols_keep.append(c)

    out = df[cols_keep].copy()

    # Drop empty unnamed columns
    out = out.loc[:, ~out.columns.astype(str).str.contains("^unnamed", case=False, na=False)]
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw registry Excel (patient-level)")
    ap.add_argument("--output", required=True, help="Path to output Excel (for pipeline)")
    ap.add_argument("--country", default="Tunisia", help="Default country value")
    ap.add_argument("--time-window", default="unknown", help="Time window label (e.g., 2018-2024)")
    ap.add_argument("--sheet", default=None, help="Optional: specific sheet name to read")
    ap.add_argument("--include-sanitized", action="store_true", help="Add a 'patients_sanitized' sheet (no identifiers)")
    args = ap.parse_args()

    # read excel
    if args.sheet:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    else:
        # default: first sheet
        df = pd.read_excel(args.input)

    # compute views
    views_df = compute_views(df, country=args.country, time_window=args.time_window)

    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        views_df.to_excel(w, sheet_name="views", index=False)

        if args.include_sanitized:
            san = sanitized_patients(df)
            san.to_excel(w, sheet_name="patients_sanitized", index=False)

    print(f"[OK] Written: {args.output}")
    print(f" - views rows: {len(views_df)}")
    if args.include_sanitized:
        print(" - patients_sanitized included")


if __name__ == "__main__":
    main()
