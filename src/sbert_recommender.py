import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_by_movie_sbert(title: str, df: pd.DataFrame, sbert_sim: np.ndarray, n: int = 5):
    if title not in df["title"].values:
        return f"'{title}' dataset mein nahi mili."

    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(sbert_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n + 1]  # skip itself

    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]

    result = df[["title", "listed_in"]].iloc[movie_indices].copy()
    result["similarity_score"] = scores
    return result


def recommend_by_query_sbert(query_text: str, model, df: pd.DataFrame, sbert_embeddings: np.ndarray, n: int = 5):
    query_vector = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    sim_scores = cosine_similarity(query_vector, sbert_embeddings)[0]

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:n]

    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]

    result = df[["title", "listed_in"]].iloc[movie_indices].copy()
    result["similarity_score"] = scores
    return result
