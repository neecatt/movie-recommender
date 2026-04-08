from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.evaluation.metrics import ndcg_at_k_weighted, precision_at_k, recall_at_k
from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
    classify_pair,
)
from src.models.hybrid import HybridRecommender


def _fresh_metrics() -> dict[str, float]:
    return {
        "top1_hit": 0.0,
        "precision_3": 0.0,
        "recall_3": 0.0,
        "ndcg_3": 0.0,
        "ndcg_10": 0.0,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    artifact_path = project_root / "models" / "hybrid_artifacts.joblib"
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}. Run scripts/train.py first.")

    df = pd.read_csv(processed_path)
    artifacts = joblib.load(artifact_path)
    base_model = HybridRecommender.from_artifacts(artifacts)

    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "90"))
    pair_queries = build_pair_queries(
        df,
        sample_size=min(sample_size, len(df)),
        random_state=42,
        min_shared_recommendations=1,
    )
    if not pair_queries:
        raise RuntimeError("No pair queries could be built from TMDB recommendations.")

    grid = []
    for content_weight in [0.15, 0.2, 0.25]:
        for embedding_weight in [0.2, 0.25, 0.3]:
            for genre_weight in [0.2, 0.25, 0.3]:
                for popularity_weight in [0.05, 0.1]:
                    for bm25_weight in [0.0]:
                        for min_votes in [300, 500, 800]:
                            grid.append(
                                (
                                    content_weight,
                                    embedding_weight,
                                    popularity_weight,
                                    bm25_weight,
                                    genre_weight,
                                    min_votes,
                                )
                            )

    best = {"ndcg_3": -1.0}

    def eval_once(
        content_weight: float,
        embedding_weight: float,
        popularity_weight: float,
        bm25_weight: float,
        genre_weight: float,
        min_votes: int,
    ) -> dict[str, object]:
        model = HybridRecommender.from_artifacts(artifacts)
        model.set_weights(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes)
        metrics = _fresh_metrics()
        slice_metrics = {
            "similar_taste": _fresh_metrics(),
            "mixed_taste": _fresh_metrics(),
            "far_apart": _fresh_metrics(),
        }
        slice_counts = {key: 0 for key in slice_metrics}

        for count, (idx_a, idx_b) in enumerate(pair_queries, start=1):
            movie_id_a = str(df.iloc[idx_a]["movie_id"])
            movie_id_b = str(df.iloc[idx_b]["movie_id"])
            gains = build_pair_relevance_gains(
                idx_a=idx_a,
                idx_b=idx_b,
                df=df,
                recommendation_sets=recommendation_sets,
                id_to_index=id_to_index,
            )
            relevant = {movie_id for movie_id, score in gains.items() if score >= 1.0}
            response = model.recommend_date_movie(movie_id_a, movie_id_b, alternatives_n=3)
            rec_ids = [response["best_pick"]["movie_id"]] + [rec["movie_id"] for rec in response["alternatives"]]

            current = {
                "top1_hit": 1.0 if rec_ids and rec_ids[0] in relevant else 0.0,
                "precision_3": precision_at_k(rec_ids, relevant, 3),
                "recall_3": recall_at_k(rec_ids, relevant, 3),
                "ndcg_3": ndcg_at_k_weighted(rec_ids, gains, 3),
                "ndcg_10": ndcg_at_k_weighted(rec_ids, gains, 10),
            }
            for key, value in current.items():
                metrics[key] += value

            pair_type = classify_pair(df, idx_a, idx_b)
            slice_counts[pair_type] += 1
            for key, value in current.items():
                slice_metrics[pair_type][key] += value

            if count % 10 == 0:
                print(f"Eval progress: {count}/{len(pair_queries)}")

        for key in metrics:
            metrics[key] /= len(pair_queries)

        for pair_type, values in slice_metrics.items():
            count = max(slice_counts[pair_type], 1)
            for key in values:
                values[key] /= count

        return {
            **metrics,
            "pair_slices": slice_metrics,
            "pair_slice_counts": slice_counts,
        }

    for content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes in grid:
        metrics = eval_once(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes)
        if float(metrics["ndcg_3"]) > best.get("ndcg_3", -1):
            best = {
                **metrics,
                "content_weight": content_weight,
                "embedding_weight": embedding_weight,
                "popularity_weight": popularity_weight,
                "bm25_weight": bm25_weight,
                "genre_weight": genre_weight,
                "min_votes": min_votes,
                "artifact_version": artifacts["artifact_version"],
                "feature_schema_version": artifacts["feature_schema_version"],
                "reranker_feature_count": artifacts["reranker_n_features"],
            }

    output_path = reports_dir / "offline_metrics.json"
    output_path.write_text(json.dumps(best, indent=2))
    print(f"Saved metrics to: {output_path}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("movie_recommender")
    with mlflow.start_run(run_name="hybrid_eval"):
        mlflow.log_metrics(
            {
                "top1_hit": float(best["top1_hit"]),
                "precision_3": float(best["precision_3"]),
                "recall_3": float(best["recall_3"]),
                "ndcg_3": float(best["ndcg_3"]),
                "ndcg_10": float(best["ndcg_10"]),
            }
        )
        mlflow.log_params(
            {
                "content_weight": best["content_weight"],
                "embedding_weight": best["embedding_weight"],
                "popularity_weight": best["popularity_weight"],
                "bm25_weight": best["bm25_weight"],
                "genre_weight": best["genre_weight"],
                "min_votes": best["min_votes"],
                "artifact_version": best["artifact_version"],
                "feature_schema_version": best["feature_schema_version"],
            }
        )


if __name__ == "__main__":
    main()
