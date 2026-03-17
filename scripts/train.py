from __future__ import annotations

from pathlib import Path
import time

import joblib
import mlflow
import pandas as pd
import os

from src.models.hybrid import HybridRecommender
from src.models.reranker import train_reranker


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
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI")
    )
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
            }
        )
        print("Fitting hybrid model (this can take a few minutes)...")
        start = time.time()
        model.fit(df)
        print(f"Model fit completed in {time.time() - start:.1f}s")

        print("Training reranker...")
        reranker = train_reranker(df, model)
        model.set_reranker(reranker)
        print("Reranker trained.")

        artifacts = model.export_artifacts()
        artifacts["training_rows"] = len(df)
        artifacts["processed_path"] = str(processed_path)
        artifact_path = models_dir / "hybrid_artifacts.joblib"
        joblib.dump(artifacts, artifact_path)
        mlflow.log_artifact(str(artifact_path))
        print(f"Saved artifacts to: {artifact_path}")


if __name__ == "__main__":
    main()
