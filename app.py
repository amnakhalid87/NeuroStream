import os
import joblib
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

from src.load_data import load_data
from src.preprocess import clean_data, build_content_column
from src.similarity import get_embeddings, get_sbert, get_similarity, get_tfidf
from src.tfidf_recommender import recommend_by_movie, recommend_by_query
from src.sbert_recommender import recommend_by_movie_sbert, recommend_by_query_sbert

RAW_DATA_PATH = r"data\netflix_data.csv.zip"
MODELS_DIR = "models"
HF_REPO = "amnaakhalid1/Netflix_data"


def get_file(filename):
    """HF dataset se file download karne ka simple function"""
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")
    except Exception:
        return os.path.join(MODELS_DIR, filename)


def setup():
    df = build_content_column(clean_data(load_data(RAW_DATA_PATH)))
    os.makedirs(MODELS_DIR, exist_ok=True)

    # TF-IDF Setup
    tfidf_file = get_file("tfidf_vectorizer.pkl")
    sim_file = get_file("tfidf_similarity.npy")

    if os.path.exists(tfidf_file) and os.path.exists(sim_file):
        tfidf = joblib.load(tfidf_file)
        tfidf_matrix = joblib.load(get_file("tfidf_matrix.pkl"))
        tfidf_sim = np.load(sim_file)
    else:
        tfidf, tfidf_matrix = get_tfidf(df["content"])
        tfidf_sim = get_similarity(tfidf_matrix)
        joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
        np.save(os.path.join(MODELS_DIR, "tfidf_similarity.npy"), tfidf_sim)

    # SBERT Setup
    sbert_model = get_sbert()
    emb_file = get_file("sbert_embeddings.npy")
    sbert_sim_file = get_file("sbert_similarity.npy")

    if os.path.exists(emb_file) and os.path.exists(sbert_sim_file):
        sbert_embeddings = np.load(emb_file)
        sbert_sim = np.load(sbert_sim_file)
    else:
        sbert_embeddings = get_embeddings(sbert_model, df["content"].tolist())
        sbert_sim = get_similarity(sbert_embeddings)
        np.save(os.path.join(MODELS_DIR, "sbert_embeddings.npy"), sbert_embeddings)
        np.save(os.path.join(MODELS_DIR, "sbert_similarity.npy"), sbert_sim)

    return df, tfidf, tfidf_matrix, tfidf_sim, sbert_model, sbert_embeddings, sbert_sim


DF, TFIDF, TFIDF_MATRIX, TFIDF_SIM, SBERT_MODEL, SBERT_EMBEDDINGS, SBERT_SIM = setup()


# Recommender Helper Functions
def tfidf_movie(title, n=5):
    res = recommend_by_movie(title, DF, TFIDF_SIM, n=int(n))
    return res if isinstance(res, str) else res.to_dict(orient="records")


def tfidf_query(query, n=5):
    return recommend_by_query(query, DF, TFIDF, TFIDF_MATRIX, n=int(n)).to_dict(orient="records")


def sbert_movie(title, n=5):
    res = recommend_by_movie_sbert(title, DF, SBERT_SIM, n=int(n))
    return res if isinstance(res, str) else res.to_dict(orient="records")


def sbert_query(query, n=5):
    return recommend_by_query_sbert(query, SBERT_MODEL, DF, SBERT_EMBEDDINGS, n=int(n)).to_dict(orient="records")


# Gradio Clean Layout
with gr.Blocks(title="Movie Recommendation System") as demo:
    gr.Markdown("# 🎬 Movie Recommendation System (TF-IDF + SBERT)")

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
