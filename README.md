---
title: Movie Recommendation System
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Movie Recommendation System (Content-Based, TF-IDF)

A content-based movie recommender built on the Netflix titles dataset.
Two ways to get recommendations:
1. **By movie title** — item-to-item similarity
2. **By free-text query** — describe what you want to watch

## Project structure

```
movie-recommendation-system/
├── data/
│   ├── raw/            # original dataset (not tracked in git)
│   └── processed/       # cleaned data (optional, generated)
├── notebooks/
│   └── movierecommendation.ipynb   # exploration / EDA / walkthrough
├── src/
│   ├── load_data.py           # loads raw CSV
│   ├── preprocess.py          # cleaning + content column
│   ├── similarity.py          # TF-IDF + cosine similarity math
│   └── tfidf_recommender.py   # recommend_by_movie, recommend_by_query
├── models/               # saved TF-IDF vectorizer, matrix, similarity (generated)
├── tests/                # unit tests
├── main.py               # runs full pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/netflix_data.csv.zip`.

## Run

```bash
python main.py
```

This will:
1. Load and clean the data
2. Build the TF-IDF matrix and similarity matrix
3. Save the trained artifacts to `models/`
4. Print a demo recommendation

## Notebook

`notebooks/movierecommendation.ipynb` is kept for exploration and explanation
only — it is not part of the production pipeline. The actual reusable logic
lives in `src/`.
