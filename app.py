import os
import joblib
import numpy as np
import gradio as gr

from src.load_data import load_data
from src.preprocess import clean_data, build_content_column
from src.similarity import get_embeddings, get_sbert, get_similarity, get_tfidf
from src.tfidf_recommender import recommend_by_movie, recommend_by_query
from src.sbert_recommender import recommend_by_movie_sbert, recommend_by_query_sbert

RAW_DATA_PATH = r"data\netflix_data.csv.zip"
MODELS_DIR = "models"


def setup():
    df = build_content_column(clean_data(load_data(RAW_DATA_PATH)))
    os.makedirs(MODELS_DIR, exist_ok=True)

    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    tfidf_matrix_path = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
    tfidf_sim_path = os.path.join(MODELS_DIR, "tfidf_similarity.npy")
    sbert_emb_path = os.path.join(MODELS_DIR, "sbert_embeddings.npy")
    sbert_sim_path = os.path.join(MODELS_DIR, "sbert_similarity.npy")

    if os.path.exists(tfidf_path) and os.path.exists(tfidf_sim_path):
        tfidf = joblib.load(tfidf_path)
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        tfidf_sim = np.load(tfidf_sim_path)
    else:
        tfidf, tfidf_matrix = get_tfidf(df["content"])
        tfidf_sim = get_similarity(tfidf_matrix)
        joblib.dump(tfidf, tfidf_path)
        joblib.dump(tfidf_matrix, tfidf_matrix_path)
        np.save(tfidf_sim_path, tfidf_sim)

    sbert_model = get_sbert()
    if os.path.exists(sbert_emb_path) and os.path.exists(sbert_sim_path):
        sbert_embeddings = np.load(sbert_emb_path)
        sbert_sim = np.load(sbert_sim_path)
    else:
        sbert_embeddings = get_embeddings(sbert_model, df["content"].tolist())
        sbert_sim = get_similarity(sbert_embeddings)
        np.save(sbert_emb_path, sbert_embeddings)
        np.save(sbert_sim_path, sbert_sim)

    return df, tfidf, tfidf_matrix, tfidf_sim, sbert_model, sbert_embeddings, sbert_sim


DF, TFIDF, TFIDF_MATRIX, TFIDF_SIM, SBERT_MODEL, SBERT_EMBEDDINGS, SBERT_SIM = setup()


def tfidf_movie(title: str, n: int = 5):
    res = recommend_by_movie(title, DF, TFIDF_SIM, n=int(n))
    return res if isinstance(res, str) else res.to_dict(orient="records")


def tfidf_query(query: str, n: int = 5):
    return recommend_by_query(query, DF, TFIDF, TFIDF_MATRIX, n=int(n)).to_dict(orient="records")


def sbert_movie(title: str, n: int = 5):
    res = recommend_by_movie_sbert(title, DF, SBERT_SIM, n=int(n))
    return res if isinstance(res, str) else res.to_dict(orient="records")


def sbert_query(query: str, n: int = 5):
    return recommend_by_query_sbert(query, SBERT_MODEL, DF, SBERT_EMBEDDINGS, n=int(n)).to_dict(orient="records")


with gr.Blocks(title="Movie Recommendation System") as demo:
    gr.Markdown("#  Movie Recommendation System (TF-IDF + SBERT)")

    tabs = [
        ("TF-IDF - By Movie", tfidf_movie, "Movie title", "Inception"),
        ("TF-IDF - By Query", tfidf_query, "Describe what you want to watch", None),
        ("SBERT - By Movie", sbert_movie, "Movie title", "Inception"),
        ("SBERT - By Query (Semantic)", sbert_query, "Describe what you want to watch", None),
    ]

    for label, func, input_label, placeholder in tabs:
        with gr.Tab(label):
            txt = gr.Textbox(label=input_label, placeholder=placeholder)
            num = gr.Number(label="Number of recommendations", value=5)
            out = gr.JSON(label="Recommendations")
            gr.Button("Recommend").click(func, inputs=[txt, num], outputs=out)

if __name__ == "__main__":
    demo.launch()
