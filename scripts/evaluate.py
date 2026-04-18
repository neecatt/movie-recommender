from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.evaluation.metrics import ndcg_at_k_weighted, precision_at_k, recall_at_k
from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
    classify_pair,
)
from src.models.hybrid import HybridRecommender
from src.models.reranker import build_features


def _fresh_metrics() -> dict[str, float]:
    return {
        "top1_hit": 0.0,
        "precision_3": 0.0,
        "recall_3": 0.0,
        "ndcg_3": 0.0,
        "ndcg_10": 0.0,
    }


def _parse_args(project_root: Path) -> argparse.Namespace:
    default_processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    parser = argparse.ArgumentParser(description="Evaluate the movie recommender model.")
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=default_processed_path,
        help=f"Path to the processed movies CSV. Default: {default_processed_path}",
    )
    return parser.parse_args()


def _build_grid() -> list[dict[str, float | int]]:
    grid: list[dict[str, float | int]] = []
    for content_weight in [0.15, 0.2, 0.25]:
        for embedding_weight in [0.2, 0.25, 0.3]:
            for genre_weight in [0.2, 0.25, 0.3]:
                for popularity_weight in [0.05, 0.1]:
                    for bm25_weight in [0.0]:
                        for min_votes in [300, 500, 800]:
                            total = (
                                content_weight
                                + embedding_weight
                                + popularity_weight
                                + bm25_weight
                                + genre_weight
                            )
                            grid.append(
                                {
                                    "content_weight": content_weight / total,
                                    "embedding_weight": embedding_weight / total,
                                    "popularity_weight": popularity_weight / total,
                                    "bm25_weight": bm25_weight / total,
                                    "genre_weight": genre_weight / total,
                                    "min_votes": min_votes,
                                }
                            )
    return grid


def _minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo + 1e-9)


def _component_scores(model: HybridRecommender, idx_list: list[int]) -> dict[str, np.ndarray]:
    if model._df is None or model._tfidf_matrix is None:
        raise RuntimeError("Model artifacts are not loaded.")

    tfidf_sims = [linear_kernel(model._tfidf_matrix[idx], model._tfidf_matrix).flatten() for idx in idx_list]
    tfidf_sim = model._aggregate_multi_seed(tfidf_sims).astype(np.float32, copy=False)

    embed_sim = np.zeros_like(tfidf_sim, dtype=np.float32)
    if model._embedding_matrix is not None:
        embed_sims = [
            cosine_similarity(model._embedding_matrix[idx].reshape(1, -1), model._embedding_matrix).flatten()
            for idx in idx_list
        ]
        embed_sim = model._aggregate_multi_seed(embed_sims).astype(np.float32, copy=False)

    genre_sim = model._genre_similarity(idx_list).astype(np.float32, copy=False)
    pop_scores = model._popularity_score().astype(np.float32, copy=False)
    bm25_scores = np.zeros_like(tfidf_sim, dtype=np.float32)
    if model._bm25 is not None and model._tokenized is not None:
        bm25_sims = [model._bm25.get_scores(model._tokenized[idx]) for idx in idx_list]
        bm25_scores = model._aggregate_multi_seed(bm25_sims).astype(np.float32, copy=False)

    return {
        "tfidf": _minmax(tfidf_sim),
        "embed": _minmax(embed_sim) if float(embed_sim.max()) > float(embed_sim.min()) else embed_sim,
        "genre": _minmax(genre_sim),
        "pop": _minmax(pop_scores),
        "bm25": _minmax(bm25_scores) if float(bm25_scores.max()) > float(bm25_scores.min()) else bm25_scores,
    }


def _combine_components(
    components: dict[str, np.ndarray],
    weights: dict[str, float | int],
) -> np.ndarray:
    return (
        float(weights["content_weight"]) * components["tfidf"]
        + float(weights["embedding_weight"]) * components["embed"]
        + float(weights["genre_weight"]) * components["genre"]
        + float(weights["popularity_weight"]) * components["pop"]
        + float(weights["bm25_weight"]) * components["bm25"]
    )


def _pair_feature_stats(
    sim_a_scores: np.ndarray,
    sim_b_scores: np.ndarray,
    joint_scores: np.ndarray,
    pair_scores: np.ndarray,
    cand_idx: int,
) -> dict[str, float]:
    sim_a = float(sim_a_scores[cand_idx])
    sim_b = float(sim_b_scores[cand_idx])
    joint_score = float(joint_scores[cand_idx])
    pair_score = float(pair_scores[cand_idx])
    return {
        "sim_a": sim_a,
        "sim_b": sim_b,
        "sim_min": min(sim_a, sim_b),
        "sim_mean": (sim_a + sim_b) / 2.0,
        "sim_gap": abs(sim_a - sim_b),
        "joint_score": joint_score,
        "pair_score": pair_score,
    }


def _candidate_indices(
    pair_data: dict[str, Any],
    weights: dict[str, float | int],
    votes: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray]:
    idx_a = pair_data["idx_a"]
    idx_b = pair_data["idx_b"]
    sim_a = _combine_components(pair_data["sim_a_components"], weights)
    sim_b = _combine_components(pair_data["sim_b_components"], weights)
    joint = _combine_components(pair_data["joint_components"], weights)
    bridge = pair_data["bridge"]
    pair_scores = (
        0.35 * np.minimum(sim_a, sim_b)
        + 0.20 * joint
        + 0.15 * ((sim_a + sim_b) / 2.0)
        + 0.30 * bridge
        - 0.10 * np.abs(sim_a - sim_b)
    )

    min_votes = int(weights["min_votes"])
    mask = votes >= min_votes
    exclusions = {idx_a, idx_b}
    faiss_candidates = pair_data["faiss_candidates"]
    top_pool = 180
    per_seed_pool = max(top_pool, 120)

    candidate_pool: set[int] = set()
    for scores in (sim_a, sim_b, joint, bridge, pair_scores):
        ranked = scores.argsort()[::-1]
        count = 0
        for idx in ranked:
            idx_int = int(idx)
            if idx_int in exclusions or not mask[idx_int]:
                continue
            if faiss_candidates and idx_int not in faiss_candidates:
                continue
            candidate_pool.add(idx_int)
            count += 1
            if count >= per_seed_pool:
                break

    ordered = sorted(candidate_pool, key=lambda idx: float(pair_scores[idx]), reverse=True)
    return pair_scores, ordered[:80], sim_a, sim_b, joint


def _recommend_ids_for_pair(
    model: HybridRecommender,
    pair_data: dict[str, Any],
    weights: dict[str, float | int],
    votes: np.ndarray,
) -> list[str]:
    if model._df is None:
        raise RuntimeError("Model artifacts are not loaded.")

    pair_scores, candidates, sim_a, sim_b, joint = _candidate_indices(pair_data, weights, votes)
    base_a = pair_data["base_a"]
    base_b = pair_data["base_b"]
    reranked: list[tuple[int, float]] = []

    for idx in candidates:
        row = model._df.iloc[idx]
        pair_stats = _pair_feature_stats(sim_a, sim_b, joint, pair_scores, idx)
        rule_score = model._paired_overlap_bonus([base_a, base_b], row)
        final_score = float(pair_stats["pair_score"]) + rule_score
        if model._reranker is not None:
            features = build_features([base_a, base_b], row, pair_stats)
            final_score += float(model._reranker.predict([features])[0])
        reranked.append((idx, final_score))

    reranked.sort(key=lambda item: item[1], reverse=True)

    genre_counts: dict[str, int] = {}
    final_ids: list[str] = []
    for idx, score in reranked:
        row = model._df.iloc[idx]
        primary_genre = str(row.get("genre", "")).split(",")[0].strip()
        adjusted_score = score - 0.05 * genre_counts.get(primary_genre, 0)
        genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1
        final_ids.append(str(row.get("movie_id", "")))
        if len(final_ids) >= 4:
            break

    return final_ids


def _prepare_pairs(
    model: HybridRecommender,
    df: pd.DataFrame,
    pair_queries: list[tuple[int, int]],
    recommendation_sets: list[set[str]],
    id_to_index: dict[str, int],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for count, (idx_a, idx_b) in enumerate(pair_queries, start=1):
        gains = build_pair_relevance_gains(
            idx_a=idx_a,
            idx_b=idx_b,
            df=df,
            recommendation_sets=recommendation_sets,
            id_to_index=id_to_index,
        )
        prepared.append(
            {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "movie_id_a": str(df.iloc[idx_a]["movie_id"]),
                "movie_id_b": str(df.iloc[idx_b]["movie_id"]),
                "pair_type": classify_pair(df, idx_a, idx_b),
                "gains": gains,
                "relevant": {movie_id for movie_id, score in gains.items() if score >= 1.0},
                "base_a": model._df.iloc[idx_a],
                "base_b": model._df.iloc[idx_b],
                "sim_a_components": _component_scores(model, [idx_a]),
                "sim_b_components": _component_scores(model, [idx_b]),
                "joint_components": _component_scores(model, [idx_a, idx_b]),
                "bridge": model._pair_bridge_scores(idx_a, idx_b).astype(np.float32, copy=False),
                "faiss_candidates": model._faiss_candidates([idx_a, idx_b]),
            }
        )
        if count % 10 == 0:
            print(f"Prepared pair caches: {count}/{len(pair_queries)}")
    return prepared


def _evaluate_grid(
    model: HybridRecommender,
    prepared_pairs: list[dict[str, Any]],
    grid: list[dict[str, float | int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if model._df is None:
        raise RuntimeError("Model artifacts are not loaded.")

    votes = model._df.get("votes", pd.Series(0, index=model._df.index)).fillna(0).to_numpy()
    results: list[dict[str, Any]] = []
    best: dict[str, Any] = {"ndcg_3": -1.0}

    for grid_index, weights in enumerate(grid, start=1):
        metrics = _fresh_metrics()
        slice_metrics = {
            "similar_taste": _fresh_metrics(),
            "mixed_taste": _fresh_metrics(),
            "far_apart": _fresh_metrics(),
        }
        slice_counts = {key: 0 for key in slice_metrics}

        for pair_data in prepared_pairs:
            rec_ids = _recommend_ids_for_pair(model, pair_data, weights, votes)
            relevant = pair_data["relevant"]
            gains = pair_data["gains"]
            current = {
                "top1_hit": 1.0 if rec_ids and rec_ids[0] in relevant else 0.0,
                "precision_3": precision_at_k(rec_ids, relevant, 3),
                "recall_3": recall_at_k(rec_ids, relevant, 3),
                "ndcg_3": ndcg_at_k_weighted(rec_ids, gains, 3),
                "ndcg_10": ndcg_at_k_weighted(rec_ids, gains, 10),
            }
            for key, value in current.items():
                metrics[key] += value

            pair_type = pair_data["pair_type"]
            slice_counts[pair_type] += 1
            for key, value in current.items():
                slice_metrics[pair_type][key] += value

        for key in metrics:
            metrics[key] /= len(prepared_pairs)

        for pair_type, values in slice_metrics.items():
            count = max(slice_counts[pair_type], 1)
            for key in values:
                values[key] /= count

        result = {
            **metrics,
            "pair_slices": slice_metrics,
            "pair_slice_counts": slice_counts,
            **weights,
        }
        results.append(result)
        if float(result["ndcg_3"]) > float(best.get("ndcg_3", -1.0)):
            best = result
        print(
            f"Grid {grid_index}/{len(grid)} | "
            f"ndcg@3={result['ndcg_3']:.4f} precision@3={result['precision_3']:.4f} "
            f"recall@3={result['recall_3']:.4f}"
        )

    return best, results


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = _parse_args(project_root)
    processed_path = args.processed_path
    if not processed_path.is_absolute():
        processed_path = (project_root / processed_path).resolve()

    artifact_path = project_root / "models" / "hybrid_artifacts.joblib"
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}. Run scripts/train.py first.")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    start = time.time()
    df = pd.read_csv(processed_path)
    artifacts = joblib.load(artifact_path)
    model = HybridRecommender.from_artifacts(artifacts)

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

    print(f"Prepared to evaluate {len(pair_queries)} pair queries.")
    prepared_pairs = _prepare_pairs(model, df, pair_queries, recommendation_sets, id_to_index)
    grid = _build_grid()
    best, all_results = _evaluate_grid(model, prepared_pairs, grid)

    best_output = {
        **best,
        "artifact_version": artifacts["artifact_version"],
        "feature_schema_version": artifacts["feature_schema_version"],
        "reranker_feature_count": artifacts["reranker_n_features"],
        "processed_path": str(processed_path),
        "sample_size": len(prepared_pairs),
        "grid_size": len(grid),
        "runtime_sec": round(time.time() - start, 3),
    }
    all_output = {
        "artifact_version": artifacts["artifact_version"],
        "feature_schema_version": artifacts["feature_schema_version"],
        "reranker_feature_count": artifacts["reranker_n_features"],
        "processed_path": str(processed_path),
        "sample_size": len(prepared_pairs),
        "grid_size": len(grid),
        "runtime_sec": round(time.time() - start, 3),
        "results": all_results,
    }

    best_path = reports_dir / "offline_metrics.json"
    grid_path = reports_dir / "offline_metrics_grid.json"
    best_path.write_text(json.dumps(best_output, indent=2))
    grid_path.write_text(json.dumps(all_output, indent=2))
    print(f"Saved best metrics to: {best_path}")
    print(f"Saved full grid metrics to: {grid_path}")


if __name__ == "__main__":
    main()
