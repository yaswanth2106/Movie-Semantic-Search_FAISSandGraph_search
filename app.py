import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from graph import text_to_graph_search

index = faiss.read_index("movie_index.faiss")
df = pd.read_pickle("movie_metadata.pkl")

MODEL_NAME = "all-mpnet-base-v2"
INDEX_PATH = "movie_index.faiss"
META_PATH = "movie_metadata.pkl"
TOP_K = 5
ALPHA = 0.7   # FAISS weight
BETA = 0.3    # Graph weight

st.set_page_config(
    page_title="🎬 Movie Semantic Search",
    layout="centered"
)

st.title("🎬 Movie Semantic Search")
st.caption("Find movies using natural language (semantic search)")


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)

@st.cache_data
def load_metadata():
    return pd.read_pickle(META_PATH)

model = load_model()
index = load_index()
df = load_metadata()


def faiss_search(query, top_k=TOP_K):
    q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "title": df.iloc[idx]["original_title"],
            "score": float(score)
        })
    return results





def hybrid_merge(faiss_results, graph_results):
    combined = {}

    for r in faiss_results:
        combined[r["title"]] = {
            "faiss": r["score"],
            "graph": 0.0
        }

    for r in graph_results:
        if r["title"] not in combined:
            combined[r["title"]] = {"faiss": 0.0, "graph": 0.0}
        combined[r["title"]]["graph"] = r["score"]

    final = []
    for title, s in combined.items():
        score = ALPHA * s["faiss"] + BETA * (s["graph"]/10)
        final.append({
            "title": title,
            "score": round(score, 3),
            "faiss": round(s["faiss"], 3),
            "graph": round(s["graph"], 3)
        })

    return sorted(final, key=lambda x: x["score"], reverse=True)[:TOP_K]



query = st.text_area(
    "Describe the movie you are looking for",
    placeholder="e.g. mind bending sci fi movie by christopher nolan involving time travel",
    height=100
)



if st.button("🔍 Search"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        with st.spinner("Searching..."):
            faiss_results = faiss_search(query)
            graph_results = text_to_graph_search(query, TOP_K)
            hybrid_results = hybrid_merge(faiss_results, graph_results)

        col1, col2, col3 = st.columns(3)

        
        with col1:
            st.subheader("🔵 FAISS (Semantic)")
            for r in faiss_results:
                st.markdown(f"**{r['title']}**  \nscore: `{r['score']:.3f}`")

       
        with col2:
            st.subheader("🟢 Graph (Structured)")
            for r in graph_results:
                st.markdown(f"**{r['title']}**  \nscore: `{r['score']:.3f}`")

       
        with col3:
            st.subheader("🔥 Hybrid (Best)")
            for r in hybrid_results:
                st.markdown(
                    f"""
                    **{r['title']}**  
                    final: `{r['score']}`  
                    faiss: `{r['faiss']}` | graph: `{r['graph']}`
                    ---
                    """
                )