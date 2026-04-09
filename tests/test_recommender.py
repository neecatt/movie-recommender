from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts.serve import create_app
from src.models.hybrid import ArtifactCompatibilityError, HybridRecommender


def _toy_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": ["a1", "b1", "c1", "d1", "e1", "dup-1", "dup-2"],
            "movie_name": [
                "Action Seed",
                "Comedy Seed",
                "Bridge Pick",
                "Action Clone",
                "Comedy Clone",
                "Same Title",
                "Same Title",
            ],
            "genre": [
                "Action, Adventure",
                "Comedy, Romance",
                "Action, Comedy, Adventure",
                "Action, Thriller",
                "Comedy, Family",
                "Drama",
                "Drama, Romance",
            ],
            "description": [
                "big adventure hero story",
                "warm funny relationship story",
                "fun adventure with humor and heart",
                "violent action revenge tale",
                "family laughs and sweet humor",
                "quiet character drama",
                "quiet romantic drama",
            ],
            "keywords": [
                "hero, adventure",
                "funny, romance",
                "adventure, funny, friendship",
                "revenge, hero",
                "funny, family",
                "drama",
                "drama, romance",
            ],
            "star": [
                "Actor A, Actor X",
                "Actor B, Actor Y",
                "Actor A, Actor B",
                "Actor A",
                "Actor B",
                "Actor D",
                "Actor E",
            ],
            "rating": [7.2, 7.0, 8.5, 6.4, 6.5, 7.1, 7.2],
            "votes": [1000, 900, 2200, 700, 680, 300, 310],
            "year": [2020, 2021, 2022, 2020, 2021, 2018, 2019],
            "runtime_min": [118, 108, 112, 130, 105, 98, 100],
            "bayesian_rating": [7.1, 6.9, 8.3, 6.2, 6.3, 7.0, 7.0],
            "recency_score": [0.91, 0.94, 0.97, 0.91, 0.94, 0.88, 0.89],
            "director": ["", "", "", "", "", "", ""],
            "tmdb_recommendations": ["c1-d1", "c1-e1", "", "", "", "", ""],
        }
    )


def _fit_model() -> HybridRecommender:
    model = HybridRecommender(min_votes=0, use_embeddings=False, embedding_weight=0.0, bm25_weight=0.0)
    model.fit(_toy_movies())
    return model


def test_pair_recommendation_prefers_shared_movie():
    model = _fit_model()
    recs = model.recommend_from_two("a1", "b1", top_n=3)
    assert recs[0].movie_id == "c1"
    assert {rec.movie_id for rec in recs}.isdisjoint({"a1", "b1"})


def test_far_apart_pair_prefers_bridge_movie_over_one_sided_clones():
    model = _fit_model()
    score_bundle = model.pair_score_bundle(0, 1)
    bridge_score = float(score_bundle["pair_scores"][2])
    action_clone_score = float(score_bundle["pair_scores"][3])
    comedy_clone_score = float(score_bundle["pair_scores"][4])

    assert bridge_score > action_clone_score
    assert bridge_score > comedy_clone_score


def test_ambiguous_title_requires_movie_id():
    model = _fit_model()
    with pytest.raises(ValueError, match="Ambiguous title"):
        model.recommend("Same Title", top_n=2)


def test_artifact_validation_rejects_stale_feature_schema():
    model = _fit_model()
    artifacts = model.export_artifacts()
    artifacts["feature_schema_version"] = "old_schema"
    with pytest.raises(ArtifactCompatibilityError):
        HybridRecommender.validate_artifacts(artifacts)


def test_artifact_validation_rejects_wrong_feature_count():
    model = _fit_model()
    artifacts = model.export_artifacts()
    artifacts["reranker_n_features"] = 7
    with pytest.raises(ArtifactCompatibilityError):
        HybridRecommender.validate_artifacts(artifacts)


def test_api_returns_best_pick_shape():
    app = create_app(model=_fit_model())
    client = TestClient(app)

    response = client.get("/recommend", params={"movie_id_a": "a1", "movie_id_b": "b1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["best_pick"]["movie_id"] == "c1"
    assert "movie_a" in payload and "movie_b" in payload
    assert "explanation" in payload


def test_search_returns_duplicate_candidates_without_auto_select():
    app = create_app(model=_fit_model())
    client = TestClient(app)

    response = client.get("/search", params={"query": "Same Title"})
    assert response.status_code == 200
    ids = [item["movie_id"] for item in response.json()["results"]]
    assert "dup-1" in ids and "dup-2" in ids
