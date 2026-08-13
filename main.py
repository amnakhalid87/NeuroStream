import os
import joblib
import numpy as np

from src.load_data import load_data
from src.preprocess import clean_data, build_content_column
from src.similarity import get_embeddings, get_sbert, get_similarity, get_tfidf
from src.tfidf_recommender import recommend_by_movie, recommend_by_query
from src.sbert_recommender import recommend_by_movie_sbert, recommend_by_query_sbert

RAW_DATA_PATH = r"data\netflix_data.csv.zip"
MODELS_DIR = "models"


def main():
    df = build_content_column(clean_data(load_data(RAW_DATA_PATH)))
    os.makedirs(MODELS_DIR, exist_ok=True)

    tfidf, tfidf_matrix = get_tfidf(df["content"])
    tfidf_sim = get_similarity(tfidf_matrix)

    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
    np.save(os.path.join(MODELS_DIR, "tfidf_similarity.npy"), tfidf_sim)

    sbert_model = get_sbert()
    sbert_embeddings = get_embeddings(sbert_model, df["content"].tolist())
    sbert_sim = get_similarity(sbert_embeddings)

    np.save(os.path.join(MODELS_DIR, "sbert_embeddings.npy"), sbert_embeddings)
    np.save(os.path.join(MODELS_DIR, "sbert_similarity.npy"), sbert_sim)

    query = "space adventure with robots and aliens"

    print("TF-IDF - Movie Based:", recommend_by_movie("Inception", df, tfidf_sim, n=5))
    print("TF-IDF - Query Based:", recommend_by_query(query, df, tfidf, tfidf_matrix, n=5))
    print("SBERT - Movie Based:", recommend_by_movie_sbert("Inception", df, sbert_sim, n=5))
    print("SBERT - Query Based:", recommend_by_query_sbert(query, sbert_model, df, sbert_embeddings, n=5))


if __name__ == "__main__":
    main()
