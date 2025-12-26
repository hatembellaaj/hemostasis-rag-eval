import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from services.embedding_viz import Method, visualize_embeddings
from services.storage import Settings, ensure_dirs


st.set_page_config(page_title="Embeddings Viz — Chroma", layout="wide")
st.title("Visualiser les embeddings (PCA / UMAP)")

settings = Settings.load()
ensure_dirs(settings)

stats_dir = settings.artifacts_dir / "stats"
viz_dir = settings.artifacts_dir / "viz"


@st.cache_data(show_spinner=False)
def _load_index_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(stats_dir.glob("indexrun_*.json"), reverse=True):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["__path"] = path
            runs.append(obj)
        except Exception:
            continue
    return runs


def _parse_filter(raw: str) -> Optional[Dict[str, Any]]:
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.error(f"Filtre JSON invalide : {exc}")
        return None


def _pick_run_id(runs: list[dict[str, Any]]) -> str:
    options = ["(saisir manuellement)"] + [
        f"{r.get('index_run_id', 'inconnu')} — {r.get('created_at', '')}" for r in runs
    ]
    choice = st.selectbox("Run d'indexation (index_run_id)", options=options, index=0)
    if choice == "(saisir manuellement)":
        return st.text_input("Saisir un index_run_id", value=st.session_state.get("last_index_run_id", ""))

    idx = options.index(choice) - 1
    return runs[idx].get("index_run_id", "") if idx >= 0 else ""


def _existing_viz_files() -> list[Path]:
    return sorted(viz_dir.glob("embeddings_2d_*.png"), reverse=True)


runs = _load_index_runs()

left, right = st.columns([1.4, 1])

with left:
    st.subheader("Paramètres")
    run_id = _pick_run_id(runs)

    method: Method = st.radio("Méthode de réduction de dimension", ["pca", "umap"], index=0)
    size_by = st.text_input("Clé pour la taille des points", value="length")
    limit = st.number_input("Limiter le nombre d'embeddings (optionnel)", min_value=0, step=500, value=0)
    where_raw = st.text_area(
        "Filtre métadonnées (JSON)",
        value="",
        placeholder='{"source_type": "paper"}',
        height=120,
    )
    random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

    generate = st.button("Générer la visualisation")

    if generate:
        if not run_id.strip():
            st.error("Merci de renseigner un index_run_id.")
        else:
            where = _parse_filter(where_raw)
            if where_raw.strip() and where is None:
                st.stop()

            with st.spinner("Chargement des embeddings et génération du graphique..."):
                try:
                    output_path = visualize_embeddings(
                        index_run_id=run_id.strip(),
                        method=method,
                        where=where,
                        size_by=size_by.strip() or "length",
                        limit=int(limit) if limit > 0 else None,
                        random_state=int(random_state),
                    )
                except Exception as exc:  # pragma: no cover - UI feedback only
                    st.error(f"Erreur lors de la génération : {exc}")
                else:
                    st.session_state["last_viz_path"] = str(output_path)
                    st.success(f"PNG sauvegardé dans {output_path}")

with right:
    st.subheader("Prévisualisation")
    preview_path = st.session_state.get("last_viz_path")
    if preview_path and Path(preview_path).exists():
        st.image(preview_path, caption=Path(preview_path).name, use_column_width=True)
    else:
        st.info("Clique sur 'Générer la visualisation' pour voir le PNG ici.")

    st.divider()
    st.subheader("Visualisations existantes")
    files = _existing_viz_files()
    if files:
        labels = [f.name for f in files]
        selected = st.selectbox("Sélectionner un PNG existant", options=labels)
        picked = files[labels.index(selected)]
        st.image(str(picked), caption=picked.name, use_column_width=True)
    else:
        st.write("Aucun PNG trouvé dans artifacts/viz/. Génère une visualisation pour commencer.")
