from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from tqdm.auto import tqdm

from src.features.text_features import _resolve_embedding_device
from src.models.hybrid import HybridRecommender
from src.models.reranker import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, train_reranker


def _func_defaults(func) -> dict:
    sig = inspect.signature(func)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    print("Loading processed data...")
    df = pd.read_csv(processed_path)
    print(f"Rows loaded: {len(df)}")

    embedding_model = os.getenv("COLAB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_device = os.getenv("COLAB_EMBEDDING_DEVICE") or _resolve_embedding_device()
    use_embeddings = _env_flag("COLAB_USE_EMBEDDINGS", True)
    use_faiss = _env_flag("COLAB_USE_FAISS", False)
    use_bm25 = _env_flag("COLAB_USE_BM25", False)
    min_votes = _env_int("COLAB_MIN_VOTES", 300)
    reranker_sample_size = _env_int("COLAB_RERANKER_SAMPLE_SIZE", 120)
    reranker_top_k = _env_int("COLAB_RERANKER_TOP_K", 80)

    print(f"Embedding device: {embedding_device}")
    print(f"Embedding model: {embedding_model}")
    print(f"Use embeddings: {use_embeddings}")
    print(f"Use FAISS: {use_faiss}")
    print(f"Use BM25: {use_bm25}")

    progress = tqdm(total=4, desc="Colab training", unit="stage")

    embedding_cache = models_dir / f"embeddings_{embedding_model}.npy"
    model = HybridRecommender(
        content_weight=_env_float("COLAB_CONTENT_WEIGHT", 0.25),
        embedding_weight=_env_float("COLAB_EMBEDDING_WEIGHT", 0.30 if use_embeddings else 0.0),
        popularity_weight=_env_float("COLAB_POPULARITY_WEIGHT", 0.10),
        bm25_weight=_env_float("COLAB_BM25_WEIGHT", 0.0 if not use_bm25 else 0.05),
        genre_weight=_env_float("COLAB_GENRE_WEIGHT", 0.35),
        min_votes=min_votes,
        use_embeddings=use_embeddings,
        use_bm25=use_bm25,
        use_faiss=use_faiss,
        embedding_model=embedding_model,
        embedding_cache_path=str(embedding_cache),
        embedding_device=embedding_device,
    )
    progress.update(1)

    print("Fitting hybrid model...")
    fit_start = time.time()
    model.fit(df)
    fit_duration = time.time() - fit_start
    print(f"Model fit completed in {fit_duration:.1f}s")
    progress.update(1)

    print("Training reranker...")
    rr_defaults = _func_defaults(train_reranker)
    rr_start = time.time()
    reranker = train_reranker(
        df,
        model,
        sample_size=reranker_sample_size,
        top_k=reranker_top_k,
    )
    reranker_duration = time.time() - rr_start
    model.set_reranker(reranker)
    print(f"Reranker trained in {reranker_duration:.1f}s")
    progress.update(1)

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
        "embedding_device": model.embedding_device,
        "use_embeddings": model.use_embeddings,
        "use_bm25": model.use_bm25,
        "use_faiss": model.use_faiss,
        "faiss_top_k": model.faiss_top_k,
        "reranker_sample_size": reranker_sample_size,
        "reranker_top_k": reranker_top_k,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "reranker_feature_count": len(FEATURE_NAMES),
        "colab_optimized": True,
        "rr_default_sample_size": rr_defaults.get("sample_size"),
        "rr_default_top_k": rr_defaults.get("top_k"),
    }
    (reports_dir / "training_metrics.json").write_text(json.dumps(training_metrics, indent=2))

    artifacts = model.export_artifacts()
    artifacts["training_rows"] = len(df)
    artifacts["processed_path"] = str(processed_path)
    artifacts["colab_optimized"] = True
    artifact_path = models_dir / "hybrid_artifacts.joblib"
    joblib.dump(artifacts, artifact_path)
    print(f"Saved artifacts to: {artifact_path}")
    print(f"Saved training metrics to: {reports_dir / 'training_metrics.json'}")
    progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()
