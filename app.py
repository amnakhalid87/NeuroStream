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

# Models load ho rahe hain
DF, TFIDF, TFIDF_MATRIX, TFIDF_SIM, SBERT_MODEL, SBERT_EMBEDDINGS, SBERT_SIM = setup()

# Recommender Wrapper Functions
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

# Gradio Interface (Bina Tabs ke Single Page Dashboard)
with gr.Blocks(title="Movie Recommendation System") as demo:
    gr.Markdown("# 🎬 Movie Recommendation System Dashboard")
    gr.Markdown("Tabs ke bug ko khatam karne ke liye saari options ek hi page par columns mein set kar di hain.")

    with gr.Row():
        # LEFT COLUMN: TF-IDF Methods
        with gr.Column():
            gr.Markdown("### 🔍 TF-IDF Recommendations (Keyword Based)")
            
            gr.Markdown("**Option 1: By Movie Title**")
            t1_title = gr.Textbox(label="Movie Title", placeholder="Inception")
            t1_n = gr.Number(label="Number of recommendations", value=5)
            t1_out = gr.JSON(label="Results")
            gr.Button("Get TF-IDF Movie Recs").click(tfidf_movie, inputs=[t1_title, t1_n], outputs=t1_out)
            
            gr.Markdown("---")
            
            gr.Markdown("**Option 2: By Text Query**")
            t2_query = gr.Textbox(label="Describe what you want to watch", placeholder="e.g., space adventure")
            t2_n = gr.Number(label="Number of recommendations", value=5)
            t2_out = gr.JSON(label="Results")
            gr.Button("Get TF-IDF Query Recs").click(tfidf_query, inputs=[t2_query, t2_n], outputs=t2_out)

        # RIGHT COLUMN: SBERT Methods
        with gr.Column():
            gr.Markdown("### 🧠 SBERT Recommendations (Semantic Meaning)")
            
            gr.Markdown("**Option 1: By Movie Title**")
            s1_title = gr.Textbox(label="Movie Title", placeholder="Inception")
            s1_n = gr.Number(label="Number of recommendations", value=5)
            s1_out = gr.JSON(label="Results")
            gr.Button("Get SBERT Movie Recs").click(sbert_movie, inputs=[s1_title, s1_n], outputs=s1_out)
            
            gr.Markdown("---")
            
            gr.Markdown("**Option 2: By Text Query**")
            s2_query = gr.Textbox(label="Describe what you want to watch", placeholder="e.g., action thriller with twists")
            s2_n = gr.Number(label="Number of recommendations", value=5)
            s2_out = gr.JSON(label="Results")
            gr.Button("Get SBERT Query Recs").click(sbert_query, inputs=[s2_query, s2_n], outputs=s2_out)

if __name__ == "__main__":
    demo.launch()
