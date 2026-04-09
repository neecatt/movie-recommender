from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
)

FEATURE_SCHEMA_VERSION = "couple_reranker_v2"
FEATURE_NAMES = (
    "sim_a",
    "sim_b",
    "sim_min",
    "sim_mean",
    "sim_gap",
    "joint_score",
    "genre_overlap_total",
    "genre_bridge",
    "genre_bridge_depth",
    "keyword_overlap_total",
    "cast_overlap_total",
    "runtime_balance",
    "quality_prior",
    "recency_balance",
    "one_sided_penalty",
)


def _split_items(value: object) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _overlap_count(a: object, b: object) -> int:
    return len(_split_items(a) & _split_items(b))


def _runtime_balance(base_rows: list[pd.Series], cand_row: pd.Series) -> float:
    base_values = [
        float(row.get("runtime_min", 0) or 0)
        for row in base_rows
        if float(row.get("runtime_min", 0) or 0) > 0
    ]
    cand_runtime = float(cand_row.get("runtime_min", 0) or 0)
    if not base_values or cand_runtime <= 0:
        return 0.0
    mean_runtime = float(np.mean(base_values))
    return max(0.0, 1.0 - min(abs(mean_runtime - cand_runtime) / 120.0, 1.0))


def _quality_prior(cand_row: pd.Series) -> float:
    bayes = float(cand_row.get("bayesian_rating", 0) or 0)
    votes = float(cand_row.get("votes", 0) or 0)
    vote_signal = min(math.log1p(votes) / math.log1p(50000), 1.0) if votes > 0 else 0.0
    return 0.7 * (bayes / 10.0) + 0.3 * vote_signal


def _recency_balance(base_rows: list[pd.Series], cand_row: pd.Series) -> float:
    base_years = [
        float(row.get("year", 0) or 0)
        for row in base_rows
        if float(row.get("year", 0) or 0) > 0
    ]
    cand_year = float(cand_row.get("year", 0) or 0)
    if not base_years or cand_year <= 0:
        return 0.0
    mean_year = float(np.mean(base_years))
    return max(0.0, 1.0 - min(abs(mean_year - cand_year) / 40.0, 1.0))


def build_features(
    base_rows: list[pd.Series],
    cand_row: pd.Series,
    pair_stats: dict[str, float],
) -> list[float]:
    genre_overlap_total = 0
    keyword_overlap_total = 0
    cast_overlap_total = 0
    per_seed_genre_overlap: list[int] = []

    for base in base_rows:
        genre_overlap = _overlap_count(base.get("genre", ""), cand_row.get("genre", ""))
        genre_overlap_total += genre_overlap
        keyword_overlap_total += _overlap_count(base.get("keywords", ""), cand_row.get("keywords", ""))
        cast_overlap_total += _overlap_count(base.get("star", ""), cand_row.get("star", ""))
        per_seed_genre_overlap.append(genre_overlap)

    genre_bridge = float(all(overlap > 0 for overlap in per_seed_genre_overlap)) if per_seed_genre_overlap else 0.0
    genre_bridge_depth = float(min(per_seed_genre_overlap)) if per_seed_genre_overlap else 0.0
    runtime_balance = _runtime_balance(base_rows, cand_row)
    quality_prior = _quality_prior(cand_row)
    recency_balance = _recency_balance(base_rows, cand_row)
    one_sided_penalty = float(
        abs(pair_stats["sim_a"] - pair_stats["sim_b"])
        + max(per_seed_genre_overlap, default=0)
        - min(per_seed_genre_overlap, default=0)
    )

    return [
        pair_stats["sim_a"],
        pair_stats["sim_b"],
        pair_stats["sim_min"],
        pair_stats["sim_mean"],
        pair_stats["sim_gap"],
        pair_stats["joint_score"],
        float(genre_overlap_total),
        genre_bridge,
        genre_bridge_depth,
        float(keyword_overlap_total),
        float(cast_overlap_total),
        runtime_balance,
        quality_prior,
        recency_balance,
        one_sided_penalty,
    ]


def train_reranker(
    df: pd.DataFrame,
    model,
    sample_size: int = 240,
    top_k: int = 120,
) -> GradientBoostingRegressor:
    pair_queries = build_pair_queries(df, sample_size=sample_size, random_state=42)
    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    X: list[list[float]] = []
    y: list[float] = []

    for idx_a, idx_b in tqdm(pair_queries, desc="Reranker training", unit="pair"):
        base_a = df.loc[idx_a]
        base_b = df.loc[idx_b]
        gains = build_pair_relevance_gains(idx_a, idx_b, df, recommendation_sets, id_to_index)
        if not gains:
            continue

        score_bundle = model.pair_score_bundle(idx_a, idx_b)
        pair_scores, candidates = model.two_seed_candidate_scores(
            idx_a,
            idx_b,
            top_pool=top_k,
            score_bundle=score_bundle,
        )
        for i in candidates:
            cand = df.loc[i]
            pair_stats = model.pair_feature_stats(
                idx_a,
                idx_b,
                i,
                pair_scores=pair_scores,
                sim_a_scores=score_bundle["sim_a"],
                sim_b_scores=score_bundle["sim_b"],
                joint_scores=score_bundle["joint"],
            )
            features = build_features([base_a, base_b], cand, pair_stats)
            relevance = gains.get(str(cand.get("movie_id", "")).strip(), 0.0)
            if relevance <= 0:
                bridge_depth = features[8]
                one_sided_penalty = features[-1]
                if pair_stats["sim_min"] >= 0.18 and bridge_depth > 0:
                    relevance = 0.08 + 0.04 * pair_stats["sim_min"] + 0.03 * bridge_depth
                elif pair_stats["sim_gap"] >= 0.28 or one_sided_penalty >= 1.5:
                    relevance = -0.12 - 0.05 * min(pair_stats["sim_gap"], 1.0)
                else:
                    relevance = 0.01 * pair_stats["sim_mean"]
            X.append(features)
            y.append(relevance)

    if not X:
        raise RuntimeError("No pairwise reranker examples were generated from TMDB recommendations.")

    reranker = GradientBoostingRegressor(random_state=42)
    reranker.fit(X, y)
    return reranker
