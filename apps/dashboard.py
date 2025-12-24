import json
from pathlib import Path

import pandas as pd
import streamlit as st

from services.storage import Settings, ensure_dirs


st.set_page_config(page_title="Dashboard — Runs & Metrics", layout="wide")
st.title("Dashboard — Runs & Metrics")

settings = Settings.load()
ensure_dirs(settings)

stats_dir = settings.artifacts_dir / "stats"
runs_dir = settings.artifacts_dir / "runs"

st.subheader("Artifacts overview")
c1, c2, c3 = st.columns(3)
c1.metric("Pivot snapshots", len(list((settings.artifacts_dir / "pivot").glob("*.jsonl"))))
c2.metric("Chunk files", len(list((settings.artifacts_dir / "chunks").glob("*.jsonl"))))
c3.metric("RAG runs", len(list(runs_dir.glob("ragrun_*.json"))))

st.divider()
st.subheader("Index run stats (Part A)")
index_stat_files = sorted(stats_dir.glob("indexrun_*.json"))
rows = []
for p in index_stat_files:
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows.append(obj)

if rows:
    df = pd.DataFrame(rows).sort_values("created_at", ascending=False)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No indexrun stats yet. Run Part A indexing first.")

st.divider()
st.subheader("RAG runs (Part B)")
rag_files = sorted(runs_dir.glob("ragrun_*.json"))
rag_rows = []
for p in rag_files:
    obj = json.loads(p.read_text(encoding="utf-8"))
    flat = {
        "run_id": obj.get("run_id"),
        "created_at": obj.get("created_at"),
        "disease": obj.get("disease"),
        "search_type": obj.get("search_type"),
        "k": obj.get("k"),
        "fetch_k": obj.get("fetch_k"),
        "rerank": obj.get("rerank"),
        "score_threshold": obj.get("score_threshold"),
        "max_context_chars": obj.get("max_context_chars"),
        "attempt": obj.get("attempt"),
        "verdict": (obj.get("verification") or {}).get("verdict"),
        "reason": (obj.get("verification") or {}).get("reason"),
        "context_chars": obj.get("context_chars"),
    }
    rag_rows.append(flat)

if rag_rows:
    df2 = pd.DataFrame(rag_rows).sort_values("created_at", ascending=False)
    st.dataframe(df2, use_container_width=True)

    st.markdown("### Verdict distribution")
    vc = df2["verdict"].value_counts(dropna=False)
    st.bar_chart(vc)
else:
    st.info("No rag runs yet. Run Part B first.")
