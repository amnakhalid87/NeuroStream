---
title: NeuroStream
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

#  NeuroStream

A content-based movie recommendation engine built on the Netflix titles dataset, served as a live REST API. Supports two recommendation strategies — classic TF-IDF keyword matching and SBERT-based semantic search — so results can be compared side by side.

**Live API:** [huggingface.co/spaces/amnaakhalid1/NeuroStream](https://huggingface.co/spaces/amnaakhalid1/NeuroStream) — interactive docs at `/docs`

---

## Overview

Given a movie title or a free-text description, the system returns the most similar movies based on genre, cast, director, and plot description.

| Approach | Input | How it works |
|---|---|---|
| **TF-IDF** | Movie title or text query | Keyword/lexical matching — compares exact word overlap |
| **SBERT** | Movie title or text query | Semantic matching — compares meaning, not just wording |

## Features

- Two independent recommendation engines (TF-IDF and SBERT), each supporting both movie-based and query-based lookup
- Trained models cached and served from the Hugging Face Hub — no need to retrain on every restart
- REST API built with FastAPI, auto-generated interactive docs (`/docs`)
- CI/CD pipeline: linting and tests run automatically on every push, deployment to Hugging Face Spaces is fully automated on merge to `main`

## Tech Stack

- **Data processing:** pandas, NumPy
- **TF-IDF:** scikit-learn
- **Semantic embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **API:** FastAPI + Uvicorn
- **Hosting:** Hugging Face Spaces (Gradio SDK, ZeroGPU)
- **Dataset & model storage:** Hugging Face Hub
- **CI/CD:** GitHub Actions

## Project Structure

```
NeuroStream/
├── .github/workflows/
│   ├── ci.yml                 # lint (flake8) + tests (pytest) on push to stage
│   └── deploy.yml             # auto-deploy to Hugging Face Space on push to main
├── notebook/
│   └── movierecommendation.ipynb   # EDA and exploration (not part of the pipeline)
├── src/
│   ├── load_data.py            # loads the raw dataset
│   ├── preprocess.py           # cleaning + builds the combined "content" text field
│   ├── similarity.py           # TF-IDF vectorization + SBERT embeddings + cosine similarity
│   ├── tfidf_recommender.py    # TF-IDF: recommend_by_movie, recommend_by_query
│   └── sbert_recommender.py    # SBERT: recommend_by_movie_sbert, recommend_by_query_sbert
├── tests/
│   └── test_recommend.py       # unit tests
├── app.py                      # FastAPI app served on Hugging Face Spaces
├── main.py                     # standalone pipeline runner (for local use)
├── requirements.txt
└── pytest.ini
```

## How Data & Models Are Loaded

The dataset and trained model artifacts (TF-IDF vectorizer, SBERT embeddings, similarity matrices) are **not stored in this repository**. Instead:

1. On startup, the app downloads the dataset from a Hugging Face Dataset repo
2. If cached model files already exist there, they're loaded directly
3. Otherwise, models are built fresh from the data and saved

This keeps the Git repository lightweight and avoids committing large binary files.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/recommend/tfidf/movie?title=Inception&n=5` | TF-IDF, movie-based |
| `GET` | `/recommend/tfidf/query?q=space+adventure&n=5` | TF-IDF, free-text query |
| `GET` | `/recommend/sbert/movie?title=Inception&n=5` | SBERT, movie-based |
| `GET` | `/recommend/sbert/query?q=space+adventure&n=5` | SBERT, semantic free-text query |
| `GET` | `/docs` | Interactive Swagger UI |

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

The API will be available at `http://localhost:7860`, with interactive docs at `http://localhost:7860/docs`.

To run the standalone pipeline (build models, print sample recommendations) instead of the API:

```bash
python main.py
```

## CI/CD Pipeline

```
push to stage  →  lint (flake8) + tests (pytest)  →  merge to main  →  auto-deploy to Hugging Face Space
```

- **`stage` branch**: every push triggers linting and unit tests via GitHub Actions
- **`main` branch**: every push (typically via merged pull request) triggers an automatic deployment to the live Hugging Face Space

## Testing

```bash
pytest tests/ -v
```

## Notebook

`notebook/movierecommendation.ipynb` documents the original exploratory analysis and step-by-step development of the recommendation logic. It is kept for reference and explanation only — the production code lives entirely in `src/`.