import os
import json
from pathlib import Path

import streamlit as st

from services.storage import Settings, ensure_dirs
from graphs.rag_graph import build_rag_graph


st.set_page_config(page_title="Part B — RAG Lab", layout="wide")
st.title("Part B — RAG Lab (Retrieval → Context → Generate → Verify → Stop/Retry)")

settings = Settings.load()
ensure_dirs(settings)

left, right = st.columns([1, 2])

with left:
    st.subheader("Query & Constraints")
    question = st.text_area("Question / Task", height=120, placeholder="Ask a question or request a protocol section draft...")

    # Single-country default: Tunisia
    country = st.text_input("Country (default Tunisia)", value="Tunisia")
    disease = st.selectbox("Disease (optional)", ["(none)", "von Willebrand disease", "hemophilia"], index=0)
    disease_val = None if disease == "(none)" else disease

    st.divider()
    st.subheader("RAG configuration")
    search_type = st.selectbox("search_type", ["similarity", "mmr"], index=0)
    k = st.slider("top-k (k)", 1, 10, 4)
    fetch_k = st.slider("fetch_k", 4, 30, 12)
    rerank = st.checkbox("rerank ON (uses similarity_search_with_score)", value=False)
    score_threshold = st.text_input("score_threshold (optional)", value="")  # left as string to avoid confusion

    max_context_chars = st.slider("max_context_chars", 1000, 20000, 8000, step=500)
    max_attempts = st.slider("max_attempts (retry loop)", 1, 5, 2)

with right:
    tab1, tab2, tab3 = st.tabs(["Run", "Explain outputs", "Artifacts"])

    with tab1:
        st.markdown("### Run RAG Graph")

        if st.button("Run"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY missing. Put it in .env and restart.")
                st.stop()
            if not question.strip():
                st.error("Write a question.")
                st.stop()

            thr = None
            if score_threshold.strip():
                try:
                    thr = float(score_threshold.strip())
                except Exception:
                    st.warning("score_threshold must be a number or empty. Ignoring.")
                    thr = None

            graph = build_rag_graph()
            state_in = {
                "settings": settings,
                "question": question.strip(),
                "country": country.strip() if country.strip() else None,
                "disease": disease_val,
                "search_type": search_type,
                "k": int(k),
                "fetch_k": int(fetch_k),
                "rerank": bool(rerank),
                "score_threshold": thr,
                "max_context_chars": int(max_context_chars),
                "max_attempts": int(max_attempts),
            }

            out = graph.invoke(state_in)

            st.success("Run completed ✅")
            st.write(f"run_id: `{out.get('run_id')}`")
            st.write("attempt:", out.get("attempt"))

            st.markdown("#### Verification")
            st.json(out.get("verification", {}))

            st.markdown("#### Context (packed)")
            st.caption(f"context_chars={out.get('context_chars')}  context_words={out.get('context_words')}")
            st.code(out.get("context", "")[:12000] + ("..." if len(out.get("context", "")) > 12000 else ""))

            st.markdown("#### Answer")
            st.write(out.get("answer", ""))

    with tab2:
        st.markdown(
            """
### What do you see here?
- **[Sx] blocks** in context are the evidence snippets.
- The model must cite **[Sx]** in the answer.
- **Verification** is currently rule-based:
  - no citations → UNSUPPORTED
  - explicit missingness + citations → PARTIAL
  - citations present → SUPPORTED (weak proxy)
- **Stop / Retry**:
  - SUPPORTED → stop
  - else retry until max_attempts
  - if context empty → stop_no_context
"""
        )

    with tab3:
        st.markdown("### Persisted runs")
        runs_dir = settings.artifacts_dir / "runs"
        run_files = sorted(runs_dir.glob("ragrun_*.json"))
        st.write(f"Found {len(run_files)} run files in `{runs_dir}`")
        show = st.slider("Show last N runs", 1, min(20, len(run_files)) if run_files else 1, 5)
        for p in run_files[-show:]:
            with st.expander(p.name):
                st.json(json.loads(p.read_text(encoding="utf-8")))
