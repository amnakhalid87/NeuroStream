import os
import joblib
import numpy as np
import gradio as gr
import spaces
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from huggingface_hub import hf_hub_download

from src.load_data import load_data
from src.preprocess import clean_data, build_content_column
from src.similarity import get_embeddings, get_sbert, get_similarity, get_tfidf
from src.tfidf_recommender import recommend_by_movie, recommend_by_query
from src.sbert_recommender import recommend_by_movie_sbert, recommend_by_query_sbert

RAW_DATA_PATH = "data/netflix_data.csv.zip"
MODELS_DIR = "models"
HF_REPO = "amnaakhalid1/Netflix_data"


@spaces.GPU
def _zerogpu_warmup():
    return True


def get_file(filename):
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")
    except Exception:
        return os.path.join(MODELS_DIR, filename)


def setup():
    df = build_content_column(clean_data(load_data(RAW_DATA_PATH)))
    os.makedirs(MODELS_DIR, exist_ok=True)

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


api = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendations using TF-IDF and SBERT",
    version="1.0.0",
)


@api.get("/")
def health():
    return {"status": "ok", "message": "Movie Recommendation API is running. See /docs for all endpoints."}


@api.get("/recommend/tfidf/movie")
def tfidf_movie(title: str = Query(...), n: int = Query(5, ge=1, le=20)):
    result = recommend_by_movie(title, DF, TFIDF_SIM, n=n)
    if isinstance(result, str):
        raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient="records")


@api.get("/recommend/tfidf/query")
def tfidf_query(q: str = Query(...), n: int = Query(5, ge=1, le=20)):
    result = recommend_by_query(q, DF, TFIDF, TFIDF_MATRIX, n=n)
    return result.to_dict(orient="records")


@api.get("/recommend/sbert/movie")
def sbert_movie(title: str = Query(...), n: int = Query(5, ge=1, le=20)):
    result = recommend_by_movie_sbert(title, DF, SBERT_SIM, n=n)
    if isinstance(result, str):
        raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient="records")


@api.get("/recommend/sbert/query")
def sbert_query(q: str = Query(...), n: int = Query(5, ge=1, le=20)):
    result = recommend_by_query_sbert(q, SBERT_MODEL, DF, SBERT_EMBEDDINGS, n=n)
    return result.to_dict(orient="records")


with gr.Blocks() as demo:
    gr.Markdown("This Space serves a REST API. See `/docs` for endpoints.")

app = gr.mount_gradio_app(api, demo, path="/ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)