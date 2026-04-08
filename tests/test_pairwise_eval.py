from __future__ import annotations

import pandas as pd

from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
    classify_pair,
)


def test_pair_relevance_prioritizes_shared_tmdb_recommendations():
    df = pd.DataFrame(
        {
            "movie_id": ["1", "2", "3", "4"],
            "movie_name": ["Seed A", "Seed B", "Shared Pick", "Solo Pick"],
            "genre": ["Action, Comedy", "Action, Sci-Fi", "Action, Comedy", "Sci-Fi"],
            "rating": [7.0, 7.2, 8.0, 6.8],
            "year": [2020, 2021, 2021, 2019],
            "tmdb_recommendations": ["3-4", "3", "", ""],
        }
    )

    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    gains = build_pair_relevance_gains(0, 1, df, recommendation_sets, id_to_index)

    assert gains["3"] > gains["4"]


def test_pair_query_builder_finds_tmdb_backed_pairs():
    df = pd.DataFrame(
        {
            "movie_id": ["1", "2", "3"],
            "movie_name": ["Seed A", "Seed B", "Shared Pick"],
            "genre": ["Action", "Action", "Action"],
            "tmdb_recommendations": ["3", "3", ""],
        }
    )

    pairs = build_pair_queries(df, sample_size=1, random_state=42)

    assert pairs == [(0, 1)]


def test_classify_pair_marks_mixed_taste():
    df = pd.DataFrame(
        {
            "movie_id": ["1", "2"],
            "movie_name": ["Seed A", "Seed B"],
            "genre": ["Action, Comedy", "Comedy, Romance"],
        }
    )

    assert classify_pair(df, 0, 1) == "mixed_taste"
