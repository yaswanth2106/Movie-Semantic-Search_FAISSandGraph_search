import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


CSV_PATH = "movies.csv"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
MAX_CHARS = 800     
TOP_K = 5

import torch
print(torch.cuda.is_available())

df = pd.read_csv(CSV_PATH)

df = df[
    ["id", "original_title", "tagline",
     "overview", "keywords", "genres"]
]

df = df.fillna("")


def build_text(row):
    text = (
        f"Title: {row['original_title']}. "
        f"Tagline: {row['tagline']}. "
        f"Overview: {row['overview']}. "
        f"Keywords: {row['keywords']}. "
        f"Genres: {row['genres']}."
    )
    return text[:MAX_CHARS]

df["embed_text"] = df.apply(build_text, axis=1)


model = SentenceTransformer(MODEL_NAME)   # use if cuda_test.py is true, device = "cuda"


embeddings = model.encode(
    df["embed_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


EMBED_DIM = embeddings.shape[1]   # guarantees correct size
index = faiss.IndexFlatIP(EMBED_DIM)  
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} movies")



def search_movie(query, top_k=TOP_K):
    q_emb = model.encode(
        query,
        normalize_embeddings=True
    ).reshape(1, -1)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "title": df.iloc[idx]["original_title"],
            "score": float(score),
            "genres": df.iloc[idx]["genres"]
        })
    return results


results = search_movie(
    "emotional science fiction movie involving space travel and time"
)

for r in results:
    print(r["title"], "| score:", round(r["score"], 3))
faiss.write_index(index, "movie_index.faiss")
META_PATH = "movie_metadata.pkl"
df.drop(columns=["embed_text"]).to_pickle(META_PATH)

print("✅ Training complete. Artifacts saved.")
