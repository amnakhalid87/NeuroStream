import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def get_tfidf(content: list, max_features: int = 10000, ngram_range: tuple = (1, 2)):
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = tfidf.fit_transform(content)
    return tfidf, tfidf_matrix


def get_similarity(matrix):
    return cosine_similarity(matrix, matrix).astype(np.float32)


def get_sbert(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_embeddings(model: SentenceTransformer, content: list) -> np.ndarray:
    embeddings = model.encode(content, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    return embeddings
