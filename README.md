# 🎬 Movie Semantic Search

A **Streamlit** web application that lets you search for movies using natural language queries. It combines two powerful techniques:

- **FAISS** semantic search with sentence‑transformer embeddings for fast, high‑quality similarity matching.
- **Graph‑based structured search** that leverages relationships between movies (e.g., shared genres, actors) for additional relevance.

The results from both methods are merged into a **hybrid ranking** (weighted by `ALPHA` and `BETA`).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [File Overview](#file-overview)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [License](#license)

---

## Project Overview
This repository contains a small demo that:
1. Loads a pre‑computed FAISS index (`movie_index.faiss`).
2. Loads movie metadata (`movie_metadata.pkl`).
3. Provides a Streamlit UI (`app.py`) where users type a query.
4. Executes a **FAISS** search and a **graph** search (`graph.py`).
5. Merges the two result sets into a final ranked list.

The data (`movies.csv`) and the generated index/metadata are already bundled, so you can start searching immediately.

---

## Architecture
```
app.py          ← Streamlit UI, orchestrates searches
├─ load_model()   → SentenceTransformer (all‑mpnet‑base‑v2)
├─ load_index()   → FAISS index (movie_index.faiss)
├─ load_metadata() → Pandas DataFrame (movie_metadata.pkl)
├─ faiss_search() → semantic similarity
├─ text_to_graph_search() (graph.py) → structured graph search
└─ hybrid_merge() → combine scores (ALPHA=0.7, BETA=0.3)
```

`graph.py` implements a simple graph‑based retrieval (e.g., based on shared attributes). The exact algorithm is hidden behind the `text_to_graph_search` function.

---

## Setup & Installation
1. **Clone the repo** (already done).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install streamlit faiss-cpu pandas sentence-transformers
   ```
4. **Download the model** (the first run will auto‑download `all-mpnet-base-v2`).

> **Note**: The repository already contains the FAISS index and metadata, so no extra data preparation is required.

---

## Running the App
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually `http://localhost:8501`).

Enter a natural‑language description of a movie (e.g., "mind‑bending sci‑fi movie by Christopher Nolan involving time travel") and click **Search**. The UI will display three columns:
- **FAISS (Semantic)** – pure embedding similarity.
- **Graph (Structured)** – results from the graph search.
- **Hybrid (Best)** – the weighted combination shown as the final recommendation.

---

## File Overview
| File | Purpose |
|------|---------|
| `app.py` | Streamlit front‑end, loads model, index, metadata, runs searches, displays results. |
| `graph.py` | Implements `text_to_graph_search` – graph‑based retrieval logic. |
| `embed_vecdb.py` | Helper script for building the FAISS index and pickling metadata (used during data preparation). |
| `cuda_test.py` | Small test script for CUDA availability (not required for the app). |
| `movie_index.faiss` | Pre‑computed FAISS index of movie embeddings. |
| `movie_metadata.pkl` | Pickled Pandas DataFrame with movie details (title, genres, etc.). |
| `movies.csv` | Original CSV source of the movie dataset. |
| `README.md` | **You are reading it!** |

---

## How It Works
1. **Embedding Generation** – `embed_vecdb.py` (run separately) encodes each movie description using `SentenceTransformer(MODEL_NAME)` and stores the vectors in a FAISS index.
2. **FAISS Search** – Given a query, the app encodes it, searches the index, and returns the top‑K most similar movies with scores.
3. **Graph Search** – `text_to_graph_search` builds a lightweight graph (e.g., movies as nodes, edges based on shared attributes) and performs a similarity walk to produce another ranked list.
4. **Hybrid Merge** – Scores are combined: `final_score = ALPHA * faiss_score + BETA * (graph_score / 10)`. The division normalises the graph score to the same magnitude as FAISS.
5. **Display** – Results are shown in three Streamlit columns for easy comparison.

---

## Customization
- **Change weighting** – Edit `ALPHA` and `BETA` in `app.py` to favour semantic or graph results.
- **Adjust top‑K** – Modify `TOP_K` (default 5) to return more/less results.
- **Swap model** – Change `MODEL_NAME` to any Sentence‑Transformer model supported by `sentence_transformers`.
- **Re‑build index** – Run `embed_vecdb.py` after modifying the dataset to regenerate `movie_index.faiss` and `movie_metadata.pkl`.

---



---

*Happy searching!*
