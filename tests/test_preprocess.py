from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.preprocess import map_tmdb_df, parse_cast
from src.data.preprocess import preprocess_movies


def test_preprocess_adds_runtime_min():
    df = pd.DataFrame(
        {
            "movie_id": ["a"],
            "movie_name": ["Movie A"],
            "runtime": ["120 min"],
            "rating": ["7.5"],
            "genre": ["Action"],
            "description": ["Story"],
            "keywords": ["hero"],
            "star": ["Actor A"],
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
            "genre": ["Action", "Action"],
            "description": ["Story", "Story"],
            "keywords": ["hero", "hero"],
            "star": ["Actor A", "Actor B"],
        }
    )
    processed = preprocess_movies(df)
    assert len(processed) == 1


def test_parse_cast_preserves_hyphenated_names():
    credits = "Jason Statham-Wu Jing-Shuya Sophia Cai-Sergio Peris-Mencheta-Skyler Samuels"
    parsed = parse_cast(credits, max_cast=5)
    assert "Sergio Peris-Mencheta" in parsed
    assert "Peris, Mencheta" not in parsed


def test_preprocess_does_not_turn_nulls_into_nan_strings():
    df = pd.DataFrame(
        {
            "movie_id": ["a", "b", "c"],
            "movie_name": ["Movie A", "Movie B", "Movie C"],
            "rating": [7.5, 7.2, 7.0],
            "genre": ["Action", "Action", "Drama"],
            "description": [None, "Has description", "Another description"],
            "keywords": [None, "hero", "drama"],
            "star": ["Actor A", "Actor B", "Actor C"],
            "director": [None, None, None],
        }
    )
    processed = preprocess_movies(df)
    assert processed.loc[0, "description"] == ""
    assert processed.loc[0, "keywords"] == ""
    assert processed.loc[0, "director"] == ""


def test_runtime_imputation_falls_back_to_primary_genre_then_global():
    df = pd.DataFrame(
        {
            "movie_id": ["a", "b", "c"],
            "movie_name": ["Movie A", "Movie B", "Movie C"],
            "rating": [7.5, 7.2, 6.8],
            "genre": ["Action, Comedy", "Action, Thriller", "Drama"],
            "description": ["Story", "Story", "Story"],
            "keywords": ["hero", "hero", "drama"],
            "star": ["Actor A", "Actor B", "Actor C"],
            "runtime_min": [100, None, None],
        }
    )
    processed = preprocess_movies(df)
    assert processed.loc[1, "runtime_min"] == 100
    assert processed.loc[2, "runtime_min"] == 100


def test_preprocess_preserves_unique_movie_id():
    df = pd.DataFrame(
        {
            "movie_id": ["a", "a"],
            "movie_name": ["Movie A", "Movie A Duplicate"],
            "rating": [7.5, 7.1],
            "genre": ["Action", "Drama"],
            "description": ["Story", "Story"],
            "keywords": ["hero", "drama"],
            "star": ["Actor A", "Actor B"],
        }
    )
    try:
        preprocess_movies(df)
    except ValueError as exc:
        assert "movie_id must remain unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate movie_id validation to fail.")


def test_tmdb_mapping_marks_director_unavailable_and_handles_sparse_fields():
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Movie A", "Movie B", "Movie C"],
            "genres": ["Action-Comedy", "Action-Thriller", "Drama"],
            "original_language": ["en", "en", "en"],
            "overview": [None, "Second movie overview", "Third movie overview"],
            "popularity": [12.0, 9.0, 7.0],
            "production_companies": [None, "Studio A", "Studio B"],
            "release_date": ["2024-01-01", "2023-01-01", "2022-01-01"],
            "budget": [0, 0, 0],
            "revenue": [1000, 1200, 800],
            "runtime": [90, 95, 88],
            "status": ["Released", "Released", "Released"],
            "tagline": [None, None, None],
            "vote_average": [7.2, 6.8, 6.5],
            "vote_count": [20, 20, 20],
            "credits": [
                "Mila Davis-Kent-José Benavidez Jr.-Selenis Leyva",
                "Actor One-Actor Two",
                "Actor Three-Actor Four",
            ],
            "keywords": [None, "thriller", "drama"],
            "poster_path": [None, None, None],
            "backdrop_path": [None, None, None],
            "recommendations": [None, None, None],
        }
    )

    mapped, summary = map_tmdb_df(raw, min_votes=10)
    processed = preprocess_movies(mapped)

    assert mapped.loc[0, "star"] == "Mila Davis-Kent, José Benavidez Jr., Selenis Leyva"
    assert processed.loc[0, "director_primary"] == ""
    assert summary["director_available"] is False
    assert processed.loc[0, "description"] == ""
