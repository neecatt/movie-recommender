from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from tqdm.auto import tqdm

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
        "precision_4": 0.0,
        "recall_4": 0.0,
        "ndcg_4": 0.0,
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.getenv(name)
    if not value:
        return default
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_mlflow_tracking_uri(project_root: Path) -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    local_tracking_dir = project_root / "mlruns"
    local_tracking_dir.mkdir(parents=True, exist_ok=True)
    return local_tracking_dir.resolve().as_uri()


def _log_to_mlflow_if_enabled(project_root: Path, best: dict[str, object]) -> None:
    enabled = os.getenv("COLAB_EVAL_USE_MLFLOW", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping MLflow logging.")
        return

    tracking_uri = _resolve_mlflow_tracking_uri(project_root)
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie_recommender")
    with mlflow.start_run(run_name="hybrid_eval_colab"):
        mlflow.log_metrics(
            {
                "top1_hit": float(best["top1_hit"]),
                "precision_4": float(best["precision_4"]),
                "recall_4": float(best["recall_4"]),
                "ndcg_4": float(best["ndcg_4"]),
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    artifact_path = project_root / "models" / "hybrid_artifacts.joblib"
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. Run scripts/train_colab.py first."
        )

    print("Loading processed data and trained artifacts...")
    df = pd.read_csv(processed_path)
    artifacts = joblib.load(artifact_path)
    HybridRecommender.validate_artifacts(artifacts)
    print(f"Rows loaded: {len(df)}")

    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    sample_size = _env_int("COLAB_EVAL_SAMPLE_SIZE", 90)
    alternatives_n = _env_int("COLAB_EVAL_ALTERNATIVES_N", 3)
    pair_queries = build_pair_queries(
        df,
        sample_size=min(sample_size, len(df)),
        random_state=42,
        min_shared_recommendations=1,
    )
    if not pair_queries:
        raise RuntimeError("No pair queries could be built from TMDB recommendations.")

    print(f"Pair queries: {len(pair_queries)}")

    grid = []
    for content_weight in _env_float_list("COLAB_EVAL_CONTENT_WEIGHTS", [0.15, 0.2, 0.25]):
        for embedding_weight in _env_float_list("COLAB_EVAL_EMBEDDING_WEIGHTS", [0.2, 0.25, 0.3]):
            for genre_weight in _env_float_list("COLAB_EVAL_GENRE_WEIGHTS", [0.2, 0.25, 0.3]):
                for popularity_weight in _env_float_list("COLAB_EVAL_POPULARITY_WEIGHTS", [0.05, 0.1]):
                    for bm25_weight in _env_float_list("COLAB_EVAL_BM25_WEIGHTS", [0.0]):
                        for min_votes in [int(x) for x in _env_float_list("COLAB_EVAL_MIN_VOTES", [300, 500, 800])]:
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

    best = {"ndcg_4": -1.0}
    outer_progress = tqdm(grid, desc="Eval grid", unit="config")

    def eval_once(
        content_weight: float,
        embedding_weight: float,
        popularity_weight: float,
        bm25_weight: float,
        genre_weight: float,
        min_votes: int,
    ) -> dict[str, object]:
        model = HybridRecommender.from_artifacts(artifacts)
        model.set_weights(
            content_weight,
            embedding_weight,
            popularity_weight,
            bm25_weight,
            genre_weight,
            min_votes,
        )
        metrics = _fresh_metrics()
        slice_metrics = {
            "similar_taste": _fresh_metrics(),
            "mixed_taste": _fresh_metrics(),
            "far_apart": _fresh_metrics(),
        }
        slice_counts = {key: 0 for key in slice_metrics}

        inner_progress = tqdm(pair_queries, desc="Pairs", unit="pair", leave=False)
        for idx_a, idx_b in inner_progress:
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
            response = model.recommend_date_movie(movie_id_a, movie_id_b, alternatives_n=alternatives_n)
            rec_ids = [response["best_pick"]["movie_id"]] + [rec["movie_id"] for rec in response["alternatives"]]

            current = {
                "top1_hit": 1.0 if rec_ids and rec_ids[0] in relevant else 0.0,
                "precision_4": precision_at_k(rec_ids, relevant, alternatives_n + 1),
                "recall_4": recall_at_k(rec_ids, relevant, alternatives_n + 1),
                "ndcg_4": ndcg_at_k_weighted(rec_ids, gains, alternatives_n + 1),
            }
            for key, value in current.items():
                metrics[key] += value

            pair_type = classify_pair(df, idx_a, idx_b)
            slice_counts[pair_type] += 1
            for key, value in current.items():
                slice_metrics[pair_type][key] += value

        inner_progress.close()

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

    for content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes in outer_progress:
        metrics = eval_once(
            content_weight,
            embedding_weight,
            popularity_weight,
            bm25_weight,
            genre_weight,
            min_votes,
        )
        outer_progress.set_postfix(
            ndcg_4=f"{metrics['ndcg_4']:.4f}",
            top1=f"{metrics['top1_hit']:.4f}",
        )
        if float(metrics["ndcg_4"]) > best.get("ndcg_4", -1):
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
                "evaluation_list_size": alternatives_n + 1,
                "colab_optimized": bool(artifacts.get("colab_optimized", False)),
            }

    outer_progress.close()

    output_path = reports_dir / "offline_metrics.json"
    output_path.write_text(json.dumps(best, indent=2))
    print(f"Saved metrics to: {output_path}")
    print(json.dumps(best, indent=2))

    _log_to_mlflow_if_enabled(project_root, best)


if __name__ == "__main__":
    main()
