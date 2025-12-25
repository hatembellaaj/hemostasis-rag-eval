from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence

import chromadb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from services.storage import Settings


Method = Literal["pca", "umap"]


@dataclass
class EmbeddingRecord:
    embedding: List[float]
    metadata: Dict[str, Any]
    document: str | None


def load_embeddings(
    chroma_dir: Path,
    collection_name: str = "docs",
    where: Dict[str, Any] | None = None,
    limit: int | None = None,
) -> List[EmbeddingRecord]:
    """Load embeddings + metadata from Chroma."""

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)
    result = collection.get(
        include=["embeddings", "metadatas", "documents"],
        where=where,
        limit=limit,
    )

    embeddings = result.get("embeddings") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []

    records: List[EmbeddingRecord] = []
    for emb, meta, doc in zip(embeddings, metadatas, documents):
        records.append(EmbeddingRecord(embedding=list(emb), metadata=meta or {}, document=doc))

    if not records:
        raise ValueError("No embeddings found for the given parameters.")

    return records


def _reduce_embeddings(embeddings: np.ndarray, method: Method, random_state: int) -> np.ndarray:
    if embeddings.shape[0] < 2:
        raise ValueError("At least two embeddings are required for visualization.")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        reducer = UMAP(n_components=2, random_state=random_state)

    return reducer.fit_transform(embeddings)


def _normalize_sizes(values: Iterable[float], min_size: float = 20.0, max_size: float = 200.0) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return np.full_like(arr, (min_size + max_size) / 2)

    scaled = (arr - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)


def _size_by(records: Sequence[EmbeddingRecord], key: str) -> np.ndarray:
    if key == "length":
        lengths = [len(r.document or "") for r in records]
        return _normalize_sizes(lengths)

    values: List[float] = []
    for r in records:
        v = r.metadata.get(key)
        if isinstance(v, (int, float)):
            values.append(float(v))
        else:
            values.append(0.0)
    return _normalize_sizes(values)


def _color_labels(records: Sequence[EmbeddingRecord], color_key: str) -> tuple[list[str], dict[str, Any]]:
    labels = [(r.metadata.get(color_key) or "unknown") for r in records]
    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique))
    colors = {label: cmap(i) for i, label in enumerate(unique)}
    return labels, colors


def _plot(
    coords: np.ndarray,
    records: Sequence[EmbeddingRecord],
    output_path: Path,
    color_key: str,
    size_by_key: str,
    title: str,
) -> None:
    labels, color_map = _color_labels(records, color_key)
    sizes = _size_by(records, size_by_key)

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in color_map:
        mask = [lbl == label for lbl in labels]
        pts = coords[mask]
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=sizes[mask],
            color=color_map[label],
            alpha=0.75,
            label=label,
            edgecolors="k",
            linewidths=0.3,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title=color_key, loc="best", fontsize="small")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def visualize_embeddings(
    index_run_id: str,
    method: Method = "pca",
    where: Dict[str, Any] | None = None,
    collection_name: str = "docs",
    size_by: str = "length",
    random_state: int = 42,
    limit: int | None = None,
) -> Path:
    """Load embeddings from Chroma, project to 2D and save scatter plot."""

    settings = Settings.load()
    records = load_embeddings(settings.chroma_dir, collection_name, where=where, limit=limit)

    embeddings = np.array([r.embedding for r in records], dtype=float)
    coords = _reduce_embeddings(embeddings, method=method, random_state=random_state)

    output_dir = settings.artifacts_dir / "viz"
    output_path = output_dir / f"embeddings_2d_{index_run_id}.png"
    title = f"Embeddings ({method.upper()}) — run {index_run_id}"
    _plot(coords, records, output_path, color_key="source_type", size_by_key=size_by, title=title)

    return output_path


def _parse_where(where_str: str | None) -> Dict[str, Any] | None:
    if not where_str:
        return None
    return json.loads(where_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize embeddings stored in Chroma.")
    parser.add_argument("index_run_id", help="Identifier used to name the output PNG.")
    parser.add_argument(
        "--method",
        choices=["pca", "umap"],
        default="pca",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--where",
        help="JSON filter applied to Chroma metadata (passed to collection.get(where=...)).",
    )
    parser.add_argument(
        "--collection",
        default="docs",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--size-by",
        default="length",
        help="Metadata key used to scale point sizes ('length' for chunk length).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of embeddings to fetch.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for dimensionality reduction algorithms.",
    )

    args = parser.parse_args()

    output = visualize_embeddings(
        index_run_id=args.index_run_id,
        method=args.method,  # type: ignore[arg-type]
        where=_parse_where(args.where),
        collection_name=args.collection,
        size_by=args.size_by,
        random_state=args.random_state,
        limit=args.limit,
    )

    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
