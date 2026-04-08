from __future__ import annotations

import pandas as pd

from src.models.reranker import FEATURE_NAMES, build_features


def test_build_features_sets_bridge_only_when_candidate_matches_both():
    base_rows = [
        pd.Series({"genre": "Action", "keywords": "hero", "star": "Star A", "runtime_min": 110, "year": 2020}),
        pd.Series({"genre": "Comedy", "keywords": "funny", "star": "Star B", "runtime_min": 100, "year": 2022}),
    ]
    cand = pd.Series(
        {
            "genre": "Action, Comedy",
            "keywords": "hero, funny",
            "star": "Star A, Star B",
            "bayesian_rating": 7.5,
            "recency_score": 0.9,
            "votes": 1000,
            "runtime_min": 108,
            "year": 2021,
        }
    )

    features = build_features(
        base_rows,
        cand,
        {
            "sim_a": 0.8,
            "sim_b": 0.7,
            "sim_min": 0.7,
            "sim_mean": 0.75,
            "sim_gap": 0.1,
            "joint_score": 0.72,
        },
    )

    assert len(features) == len(FEATURE_NAMES)
    assert features[7] == 1.0
