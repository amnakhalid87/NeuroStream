import pandas as pd
from src.preprocess import trim_cast
from src.tfidf_recommender import recommend_by_movie


def test_trim_cast():
    result = trim_cast("A, B, C, D, E")
    assert result == "A B C"


def test_recommend_unknown_movie():
    df = pd.DataFrame({"title": ["A", "B"], "listed_in": ["X", "Y"]})
    sim_matrix = [[1, 0], [0, 1]]
    result = recommend_by_movie("Unknown Movie", df, sim_matrix, n=2)
    assert isinstance(result, str)
    assert "not found" in result
