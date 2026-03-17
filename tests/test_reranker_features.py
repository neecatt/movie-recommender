from __future__ import annotations

import pandas as pd

from src.models.reranker import build_features


def test_build_features_ignores_empty_director_matches():
    base = pd.Series(
        {
            "genre": "Action",
            "director_primary": "",
            "star_primary": "Star A",
        }
    )
    cand = pd.Series(
        {
            "genre": "Action",
            "director_primary": "",
            "star_primary": "Star A",
            "bayesian_rating": 7.5,
            "recency_score": 0.9,
            "votes": 1000,
        }
    )

    features = build_features(
        [base],
        cand,
        {
            "sim_a": 0.8,
            "sim_b": 0.7,
            "sim_min": 0.7,
            "sim_mean": 0.75,
            "sim_gap": 0.1,
            "joint_score": 0.72,
            "embed_a": 0.0,
            "embed_b": 0.0,
            "embed_min": 0.0,
            "bm25_a": 0.0,
            "bm25_b": 0.0,
        },
    )

    assert features[12] == 0
