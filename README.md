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

## 2) Prérequis

- Docker + Docker Compose
- Une clé OpenAI dans `.env` (NE PAS COMMIT)

Créer ton `.env` à partir de l’exemple :

```bash
cp .env.example .env
# puis éditer .env
