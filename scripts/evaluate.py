from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import mlflow

from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
)
from src.evaluation.metrics import ndcg_at_k_weighted, precision_at_k, recall_at_k
from src.models.hybrid import HybridRecommender

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    use_embeddings = os.getenv("EVAL_USE_EMBEDDINGS", "1") == "1"
    sample_size = int(os.getenv("EVAL_SAMPLE_SIZE", "100"))
    max_candidates = int(os.getenv("EVAL_MAX_CANDIDATES", "5000"))
    embedding_cache = project_root / "models" / "embeddings_all-mpnet-base-v2.npy"
    model = HybridRecommender(
        use_embeddings=use_embeddings,
        use_bm25=False,
        embedding_model="all-mpnet-base-v2",
        embedding_cache_path=str(embedding_cache),
    )
    model.fit(df)

    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    pair_queries = build_pair_queries(
        df,
        sample_size=min(sample_size, len(df)),
        random_state=42,
        min_shared_recommendations=1,
    )
    if not pair_queries:
        raise RuntimeError("No pair queries could be built from TMDB recommendations.")

    use_optuna = os.getenv("EVAL_OPTUNA", "0") == "1"
    grid = []
    for content_weight in [0.1, 0.2, 0.3]:
        for embedding_weight in [0.2, 0.3, 0.4]:
            for genre_weight in [0.2, 0.3, 0.4]:
                for popularity_weight in [0.0, 0.1]:
                    for bm25_weight in [0.0]:
                        for min_votes in [300, 500, 800]:
                            grid.append((content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes))

    best = {"ndcg_10": -1.0}
    def eval_once(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes):
        model.set_weights(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes)
        metrics = {"precision_10": 0.0, "recall_10": 0.0, "ndcg_10": 0.0}

        for count, (idx_a, idx_b) in enumerate(pair_queries, start=1):
            title_a = str(df.iloc[idx_a]["movie_name"])
            title_b = str(df.iloc[idx_b]["movie_name"])
            scores = build_pair_relevance_gains(
                idx_a=idx_a,
                idx_b=idx_b,
                df=df,
                recommendation_sets=recommendation_sets,
                id_to_index=id_to_index,
            )
            if max_candidates:
                top_labels = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_candidates]
                scores = dict(top_labels)
            relevant = {name for name, score in scores.items() if score >= 1.0}
            recs = [r.movie_name for r in model.recommend_from_two(title_a, title_b, top_n=10)]
            metrics["precision_10"] += precision_at_k(recs, relevant, 10)
            metrics["recall_10"] += recall_at_k(recs, relevant, 10)
            metrics["ndcg_10"] += ndcg_at_k_weighted(recs, scores, 10)
            if count % 10 == 0:
                print(f"Eval progress: {count}/{len(pair_queries)}")

        for key in metrics:
            metrics[key] = metrics[key] / len(pair_queries)

        return metrics

    if use_optuna:
        import optuna

        def objective(trial: optuna.Trial) -> float:
            content_weight = trial.suggest_float("content_weight", 0.05, 0.4)
            embedding_weight = trial.suggest_float("embedding_weight", 0.1, 0.5)
            genre_weight = trial.suggest_float("genre_weight", 0.1, 0.5)
            popularity_weight = trial.suggest_float("popularity_weight", 0.0, 0.2)
            bm25_weight = trial.suggest_float("bm25_weight", 0.0, 0.1)
            min_votes = trial.suggest_categorical("min_votes", [100, 300, 500, 800])
            metrics = eval_once(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes)
            trial.set_user_attr("metrics", metrics)
            return metrics["ndcg_10"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(os.getenv("EVAL_OPTUNA_TRIALS", "30")))
        best_params = study.best_params
        best_metrics = study.best_trial.user_attrs["metrics"]
        best = {**best_metrics, **best_params}
    else:
        for content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes in grid:
            metrics = eval_once(content_weight, embedding_weight, popularity_weight, bm25_weight, genre_weight, min_votes)
            if metrics["ndcg_10"] > best.get("ndcg_10", -1):
                best = {
                    **metrics,
                    "content_weight": content_weight,
                    "embedding_weight": embedding_weight,
                    "popularity_weight": popularity_weight,
                    "bm25_weight": bm25_weight,
                    "genre_weight": genre_weight,
                    "min_votes": min_votes,
                }

    output_path = reports_dir / "offline_metrics.json"
    output_path.write_text(json.dumps(best, indent=2))
    print(f"Saved metrics to: {output_path}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("movie_recommender")
    with mlflow.start_run(run_name="hybrid_eval"):
        mlflow.log_metrics(
            {
                "precision_10": best["precision_10"],
                "recall_10": best["recall_10"],
                "ndcg_10": best["ndcg_10"],
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
            }
        )


if __name__ == "__main__":
    main()
