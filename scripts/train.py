from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.models.hybrid import HybridRecommender
from src.models.reranker import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, train_reranker


def _func_defaults(func) -> dict:
    sig = inspect.signature(func)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _resolve_mlflow_tracking_uri(project_root: Path) -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    local_tracking_dir = project_root / "mlruns"
    local_tracking_dir.mkdir(parents=True, exist_ok=True)
    return local_tracking_dir.resolve().as_uri()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    print("Loading processed data...")
    df = pd.read_csv(processed_path)
    print(f"Rows loaded: {len(df)}")
    tracking_uri = _resolve_mlflow_tracking_uri(project_root)
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("movie_recommender")

    embedding_cache = models_dir / "embeddings_all-mpnet-base-v2.npy"
    model = HybridRecommender(
        content_weight=0.20,
        embedding_weight=0.30,
        popularity_weight=0.10,
        bm25_weight=0.05,
        genre_weight=0.35,
        min_votes=500,
        use_embeddings=True,
        use_bm25=False,
        use_faiss=True,
        embedding_model="all-mpnet-base-v2",
        embedding_cache_path=str(embedding_cache),
    )
    with mlflow.start_run(run_name="hybrid_train"):
        mlflow.log_params(
            {
                "content_weight": model.content_weight,
                "embedding_weight": model.embedding_weight,
                "popularity_weight": model.popularity_weight,
                "bm25_weight": model.bm25_weight,
                "genre_weight": model.genre_weight,
                "min_votes": model.min_votes,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "reranker_feature_count": len(FEATURE_NAMES),
            }
        )
        print("Fitting hybrid model (this can take a few minutes)...")
        fit_start = time.time()
        model.fit(df)
        fit_duration = time.time() - fit_start
        print(f"Model fit completed in {fit_duration:.1f}s")

        print("Training reranker...")
        rr_defaults = _func_defaults(train_reranker)
        rr_start = time.time()
        reranker = train_reranker(df, model)
        reranker_duration = time.time() - rr_start
        model.set_reranker(reranker)
        print("Reranker trained.")

        reports_dir = project_root / "reports" / "results"
        reports_dir.mkdir(parents=True, exist_ok=True)
        training_metrics = {
            "fit_duration_sec": round(fit_duration, 3),
            "reranker_fit_duration_sec": round(reranker_duration, 3),
            "training_rows": len(df),
            "content_weight": model.content_weight,
            "embedding_weight": model.embedding_weight,
            "popularity_weight": model.popularity_weight,
            "bm25_weight": model.bm25_weight,
            "genre_weight": model.genre_weight,
            "min_votes": model.min_votes,
            "embedding_model": model.embedding_model,
            "use_embeddings": model.use_embeddings,
            "use_bm25": model.use_bm25,
            "use_faiss": model.use_faiss,
            "faiss_top_k": model.faiss_top_k,
            "reranker_sample_size": rr_defaults.get("sample_size"),
            "reranker_top_k": rr_defaults.get("top_k"),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "reranker_feature_count": len(FEATURE_NAMES),
        }
        (reports_dir / "training_metrics.json").write_text(
            json.dumps(training_metrics, indent=2)
        )
        mlflow.log_metrics(
            {
                "train_fit_duration_sec": fit_duration,
                "train_reranker_duration_sec": reranker_duration,
                "train_rows": len(df),
            }
        )

        artifacts = model.export_artifacts()
        artifacts["training_rows"] = len(df)
        artifacts["processed_path"] = str(processed_path)
        artifact_path = models_dir / "hybrid_artifacts.joblib"
        joblib.dump(artifacts, artifact_path)
        mlflow.log_artifact(str(artifact_path))
        print(f"Saved artifacts to: {artifact_path}")


if __name__ == "__main__":
    main()
