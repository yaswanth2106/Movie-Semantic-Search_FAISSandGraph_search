import pandas as pd
import networkx as nx
import re
import json
import cohere


CSV_PATH = "movies.csv"   # your dataset
TOP_K = 5
COHERE_API_KEY = "key here!!!!!!!!!!!!!!!!!!!!!!!!"


co = cohere.Client(COHERE_API_KEY)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_vals(x):
    return [normalize(v) for v in x.split("|") if v.strip()]


df = pd.read_csv(CSV_PATH).fillna("")


G = nx.Graph()

for _, row in df.iterrows():
    movie_node = f"movie:{row['id']}"
    G.add_node(movie_node, type="movie", title=row["original_title"])

    for actor in split_vals(row["cast"]):
        G.add_edge(movie_node, f"actor:{actor}")

    for director in split_vals(row["director"]):
        G.add_edge(movie_node, f"director:{director}")

    for genre in split_vals(row["genres"]):
        G.add_edge(movie_node, f"genre:{genre}")

    for kw in split_vals(row["keywords"]):
        G.add_edge(movie_node, f"keyword:{kw}")

    for comp in split_vals(row["production_companies"]):
        G.add_edge(movie_node, f"company:{comp}")

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


def build_graph_vocab(G):
    vocab = {
        "actor": set(),
        "director": set(),
        "genre": set(),
        "keyword": set(),
        "company": set()
    }

    for node in G.nodes:
        if ":" in node:
            t, name = node.split(":", 1)
            if t in vocab:
                vocab[t].add(name)

    return {k: sorted(v) for k, v in vocab.items()}

GRAPH_VOCAB = build_graph_vocab(G)


EXTRACT_PROMPT = """
You are an expert movie information extractor.

From the user query, extract movie-related entities.
If something is IMPLIED, still extract it.

Return ONLY valid JSON.
Do NOT explain anything.

Use this exact JSON format:
{{
  "actors": [],
  "directors": [],
  "genres": [],
  "keywords": [],
  "production_companies": []
}}

Examples:
Query: "a sci fi movie by christopher nolan"
Output:
{{
  "actors": [],
  "directors": ["christopher nolan"],
  "genres": ["science fiction"],
  "keywords": [],
  "production_companies": []
}}

Query: "mind bending time travel movie"
Output:
{{
  "actors": [],
  "directors": [],
  "genres": [],
  "keywords": ["mind bending", "time travel"],
  "production_companies": []
}}

Now extract from this query:
"{query}"
"""



LINK_PROMPT = """
You are an entity linking system.

Given an extracted entity and a list of canonical entities,
return the SINGLE closest match.

Rules:
- Return ONLY the matched entity
- If no good match exists, return NONE
- Do NOT invent new entities

Extracted entity:
"{entity}"

Canonical entities:
{candidates}
"""

def link_entity_cohere(entity, candidates):
    if not candidates:
        return None

    response = co.chat(
        model="command-xlarge-nightly",
        message=LINK_PROMPT.format(
            entity=entity,
            candidates=", ".join(candidates[:200])
        ),
        temperature=0
    )

    result = normalize(response.text)
    if result == "none":
        return None

    if result not in candidates:
        return None
    return result

def link_entities(extracted, graph_vocab):
    linked = {
        "actors": [],
        "directors": [],
        "genres": [],
        "keywords": [],
        "production_companies": []
    }

    mapping = {
        "actors": "actor",
        "directors": "director",
        "genres": "genre",
        "keywords": "keyword",
        "production_companies": "company"
    }

    for k, vals in extracted.items():
        gtype = mapping[k]
        for v in vals:
            v = normalize(v)
            match = link_entity_cohere(v, graph_vocab[gtype])
            if match:
                linked[k].append(match)

    return linked


WEIGHTS = {
    "directors": 4.0,
    "actors": 2.0,
    "genres": 1.5,
    "keywords": 1.0,
    "production_companies": 0.5
}


EMPTY = {
    "actors": [],
    "directors": [],
    "genres": [],
    "keywords": [],
    "production_companies": []
}

def extract_entities_cohere(query):
    response = co.chat(
        model="command-xlarge-nightly",   
        temperature=0,
        preamble="""
You are a movie entity extraction system.

Extract entities from the user query.
Infer entities even if implied.

Return ONLY valid JSON.
No explanations. No markdown.

JSON format:
{
  "actors": [],
  "directors": [],
  "genres": [],
  "keywords": [],
  "production_companies": []
}

Examples:

User: a sci fi movie by christopher nolan
Output:
{
  "actors": [],
  "directors": ["christopher nolan"],
  "genres": ["science fiction"],
  "keywords": [],
  "production_companies": []
}

User: mind bending time travel movie
Output:
{
  "actors": [],
  "directors": [],
  "genres": [],
  "keywords": ["mind bending", "time travel"],
  "production_companies": []
}
""",
        message=query
    )

    raw = response.text.strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError
        return data
    except Exception:
        return {
            "actors": [],
            "directors": [],
            "genres": [],
            "keywords": [],
            "production_companies": []
        }


def text_to_graph_search(query, top_k=TOP_K):
    extracted = extract_entities_cohere(query)

    linked = link_entities(extracted, GRAPH_VOCAB)

    scores = {}

    for etype, values in linked.items():
        weight = WEIGHTS.get(etype, 0)
        for v in values:
            node = f"{etype[:-1]}:{v}"
            

            if not G.has_node(node):
                continue

            for movie in G.neighbors(node):
                scores[movie] = scores.get(movie, 0) + weight
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for movie_node, score in ranked:
        movie_id = int(movie_node.split(":")[1])
        title = df.loc[df["id"] == movie_id, "original_title"].values[0]

        results.append({
            "title": title,
            "score": round(score, 3)
        })

    return results


