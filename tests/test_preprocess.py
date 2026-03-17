from __future__ import annotations

import pandas as pd

from src.data.preprocess import preprocess_movies


def test_preprocess_adds_runtime_min():
    df = pd.DataFrame(
        {
            "movie_id": ["a"],
            "movie_name": ["Movie A"],
            "runtime": ["120 min"],
            "rating": ["7.5"],
        }
    )
    processed = preprocess_movies(df)
    assert "runtime_min" in processed.columns
    assert processed.loc[0, "runtime_min"] == 120


def test_preprocess_drops_missing_rating():
    df = pd.DataFrame(
        {
            "movie_id": ["a", "b"],
            "movie_name": ["Movie A", "Movie B"],
            "rating": ["7.5", None],
        }
    )
    processed = preprocess_movies(df)
    assert len(processed) == 1
