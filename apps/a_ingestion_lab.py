import os
from pathlib import Path

import streamlit as st

from services.storage import Settings, ensure_dirs
from graphs.ingestion_graph import build_ingestion_graph
from graphs.index_graph import build_index_graph


st.set_page_config(page_title="Part A — Ingestion Lab", layout="wide")
st.title("Part A — Ingestion Lab (PDF + Registry Excel → Pivot → Chunks → Index)")

settings = Settings.load()
ensure_dirs(settings)

left, right = st.columns([1, 2])

with left:
    st.subheader("Inputs")

    papers_dir = settings.data_papers_dir
    registry_dir = settings.data_registry_dir

    st.write(f"📁 papers dir: `{papers_dir}`")
    st.write(f"📁 registry dir: `{registry_dir}`")

    # List available papers
    paper_files = sorted([p for p in papers_dir.glob("*") if p.suffix.lower() in [".pdf", ".txt", ".md"]])
    selected_papers = st.multiselect(
        "Select paper files (from data/papers/)",
        options=[str(p) for p in paper_files],
        default=[str(paper_files[0])] if paper_files else [],
    )

    # Registry excel
    excel_files = sorted([p for p in registry_dir.glob("*.xlsx")])
    selected_excel = st.selectbox(
        "Select registry Excel (from data/registry/)",
        options=["(none)"] + [str(p) for p in excel_files],
        index=0,
    )
    excel_path = None if selected_excel == "(none)" else selected_excel

    st.divider()
    st.subheader("Ingestion variant")
    # For Part A comparison: split/chunk knobs here
    split_mode = st.selectbox("split_mode", ["page", "heading"], index=0)
    chunk_mode = st.selectbox("chunk_mode", ["fixed", "structure", "semantic"], index=0)
    chunk_size = st.slider("chunk_size", 300, 2000, 1000, step=50)
    chunk_overlap = st.slider("chunk_overlap", 0, 400, 200, step=20)

    st.divider()
    st.subheader("Persistence")
    st.write(f"🧠 Chroma dir: `{settings.chroma_dir}`")
    st.write(f"🧾 Artifacts: `{settings.artifacts_dir}`")

with right:
    st.subheader("Run & Observe")

    tab1, tab2, tab3 = st.tabs(["1) Ingestion (Pivot)", "2) Indexing (Chunks + Chroma)", "3) Tips"])

    # ---- Tab 1: ingestion
    with tab1:
        st.markdown("### Step A1 — Build Pivot Snapshot")
        st.write("Loads PDFs + optional registry Excel, outputs a pivot JSONL snapshot.")

        if st.button("Run Ingestion (build pivot snapshot)"):
            if not selected_papers and not excel_path:
                st.error("Select at least one paper OR a registry Excel.")
                st.stop()

            graph = build_ingestion_graph()
            state_in = {
                "settings": settings,
                "paper_paths": selected_papers,
                "registry_excel_path": excel_path,
            }
            out = graph.invoke(state_in)

            st.success("Ingestion completed ✅")
            st.json(out.get("stats", {}))
            st.write("Pivot path:", out.get("pivot_path"))

            st.session_state["last_pivot_path"] = out.get("pivot_path")
            st.session_state["last_snapshot_id"] = out.get("snapshot_id")

    # ---- Tab 2: index
    with tab2:
        st.markdown("### Step A2 — Chunk + Index into Chroma")
        st.write("Reads pivot snapshot, produces chunks and adds them to Chroma (persisted).")

        pivot_path = st.session_state.get("last_pivot_path")
        if pivot_path:
            st.info(f"Using last pivot: `{pivot_path}`")
        else:
            st.warning("No pivot yet. Run ingestion first, or pick an existing pivot snapshot below.")

        # allow selecting existing pivot files
        pivot_files = sorted((settings.artifacts_dir / "pivot").glob("*.jsonl"))
        selected_pivot = st.selectbox(
            "Select pivot snapshot",
            options=["(use last)"] + [str(p) for p in pivot_files],
            index=0,
        )
        if selected_pivot != "(use last)":
            pivot_path = selected_pivot

        if st.button("Run Indexing (chunk + add to Chroma)"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY missing. Put it in .env and restart.")
                st.stop()
            if not pivot_path:
                st.error("No pivot selected.")
                st.stop()

            graph = build_index_graph()
            state_in = {
                "settings": settings,
                "pivot_path": pivot_path,
                "split_mode": split_mode,
                "chunk_mode": chunk_mode,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
            }
            out = graph.invoke(state_in)

            st.success("Indexing completed ✅")
            st.json(out.get("stats", {}))
            st.write("Chunks path:", out.get("chunks_path"))
            st.write("Chroma count (approx):", out.get("chroma_count"))

            st.session_state["last_chunks_path"] = out.get("chunks_path")
            st.session_state["last_index_run_id"] = out.get("index_run_id")

    with tab3:
        st.markdown(
            """
### What to check (Part A)
- **Pivot rows**: papers pages + registry-derived rows
- **Chunk stats**: empty_rate, dup_rate, size distribution
- **Chroma count**: should increase after indexing
- **Artifacts**:
  - `artifacts/pivot/*.jsonl`
  - `artifacts/chunks/*.jsonl`
  - `artifacts/stats/*.json`
  - `chroma_db/` contains persisted files
"""
        )
