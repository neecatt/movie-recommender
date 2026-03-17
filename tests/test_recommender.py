from __future__ import annotations

import pandas as pd

from src.models.hybrid import HybridRecommender


def test_recommender_output_shape():
    df = pd.DataFrame(
        {
            "movie_id": ["a", "b", "c"],
            "movie_name": ["Movie A", "Movie B", "Movie C"],
            "genre": ["Action", "Action", "Drama"],
            "description": ["x", "y", "z"],
            "director": ["Dir1", "Dir2", "Dir3"],
            "star": ["Star1", "Star2", "Star3"],
            "rating": [7.0, 6.5, 8.0],
            "votes": [100, 120, 200],
        }
    )
    model = HybridRecommender(min_votes=0, use_embeddings=False, embedding_weight=0.0, content_weight=1.0)
    model.fit(df)
    recs = model.recommend("Movie A", top_n=2)
    assert len(recs) == 2
