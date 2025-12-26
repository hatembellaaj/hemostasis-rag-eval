# Hemostasis RAG Eval (Part A Ingestion + Part B RAG)

Ce projet est un **laboratoire expérimental** (orienté papier) pour comparer :
- **Part A — Ingestion** : extraction + normalisation + splitting/chunking → pivot + chunks
- **Part B — RAG** : retrieval (similarity/MMR), rerank/threshold, context budget, verification + stop condition

Contexte : cas d'étude sur des **maladies de l'hémostase** (ex. von Willebrand disease / hemophilia) et un registre local (Tunisie).

---

## 1) Arborescence utile

- `data/papers/` : dépose tes articles (PDF/TXT/MD)
- `data/registry/` : dépose ton Excel de registre (ex: `registry.xlsx`)
- `artifacts/pivot/` : snapshots pivot JSONL (sortie ingestion)
- `artifacts/chunks/` : chunks JSONL (sortie chunking)
- `artifacts/stats/` : stats ingestion/indexing
- `artifacts/runs/` : runs RAG (inputs/config/verdict/answer/context)
- `chroma_db/` : base Chroma persistée (embeddings + index)

---

## Visualiser les embeddings (PCA / UMAP)

Une fois l'indexation Chroma effectuée, tu peux générer un scatterplot 2D des embeddings :

```bash
# activer ton venv et installer les dépendances si besoin
pip install -r requirements.txt

# générer le PNG (artifacts/viz/embeddings_2d_<run>.png)
python -m services.embedding_viz <index_run_id> \
  --method umap \
  --where '{"source_type": "paper"}' \
  --size-by length \
  --limit 2000
```

Options utiles :
- `--method` : `pca` (par défaut) ou `umap`.
- `--where` : filtre JSON appliqué aux métadonnées Chroma (ex: `{"source_type": "registry_view"}`).
- `--collection` : nom de collection (par défaut `docs`).
- `--size-by` : clé métadonnée utilisée pour la taille des points (`length` = taille du chunk).
- `--limit` : limite le nombre d'embeddings chargés.
- `--random-state` : graine pour la réduction de dimension.

### Via l'interface Streamlit

Une page Streamlit dédiée permet d'explorer/générer les PNG via un bouton :

```bash
streamlit run apps/embedding_viz.py
```

Sur la page "Visualiser les embeddings" :

- choisis un `index_run_id` existant (ou saisis-le manuellement),
- sélectionne la méthode (PCA/UMAP), un éventuel filtre JSON (`where`), un `size_by` et une limite,
- clique sur **Générer la visualisation** pour produire et pré-visualiser le PNG, ou ouvrir un PNG existant depuis `artifacts/viz/`.

### Utilisation via Docker / Docker Compose

La visualisation est incluse dans l'image Docker :

```bash
# construire l'image (si nécessaire)
docker compose build

# lancer uniquement la page Streamlit de visualisation (http://localhost:8504)
docker compose up embedding_viz

# (optionnel) générer un PNG en ligne de commande depuis le conteneur
docker compose run --rm embedding_viz \
  python -m services.embedding_viz <index_run_id> --method umap --limit 2000
```

Les volumes `data/`, `artifacts/` et `chroma_db/` sont montés dans le conteneur, donc les PNG sont sauvegardés dans `artifacts/viz/` sur l'hôte.

---

## 2) Prérequis

- Docker + Docker Compose
- Une clé OpenAI dans `.env` (NE PAS COMMIT)

Créer ton `.env` à partir de l’exemple :

```bash
cp .env.example .env
# puis éditer .env
