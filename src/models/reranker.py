from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.evaluation.pairwise import (
    build_pair_queries,
    build_pair_relevance_gains,
    build_tmdb_recommendation_sets,
    classify_pair,
)

FEATURE_SCHEMA_VERSION = "couple_reranker_v4"
FEATURE_NAMES = (
    "sim_a",
    "sim_b",
    "sim_min",
    "sim_mean",
    "sim_gap",
    "joint_score",
    "utility_product",
    "utility_harmonic",
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

LABEL_STRENGTH = {
    "good_for_both": 4.0,
    "acceptable_compromise": 3.0,
    "leans_to_a": 2.0,
    "leans_to_b": 2.0,
    "bad_for_both": 1.0,
}

CONFIDENCE_MULTIPLIER = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "": 1,
}


@dataclass
class PairwiseLinearRanker:
    """Linear scorer trained on pairwise feature differences."""

    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    n_features_in_: int = 0

    def fit(self, X: list[list[float]], y: list[int]) -> "PairwiseLinearRanker":
        clf = LogisticRegression(random_state=42, max_iter=2000)
        clf.fit(X, y)
        self.coef_ = clf.coef_[0].astype(np.float32, copy=False)
        self.intercept_ = float(clf.intercept_[0])
        self.n_features_in_ = int(clf.n_features_in_)
        return self

    def predict(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr @ self.coef_ + self.intercept_


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
    utility_product = float(pair_stats["sim_a"] * pair_stats["sim_b"])
    utility_harmonic = 0.0
    if pair_stats["sim_a"] > 0 and pair_stats["sim_b"] > 0:
        utility_harmonic = float(
            2.0 * pair_stats["sim_a"] * pair_stats["sim_b"] / (pair_stats["sim_a"] + pair_stats["sim_b"] + 1e-9)
        )
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
        utility_product,
        utility_harmonic,
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
    compromise_labels_path: str | None = None,
) -> PairwiseLinearRanker:
    pair_queries = build_pair_queries(df, sample_size=sample_size, random_state=42)
    recommendation_sets, id_to_index = build_tmdb_recommendation_sets(df)
    X: list[list[float]] = []
    y: list[int] = []

    for idx_a, idx_b in tqdm(pair_queries, desc="Reranker training", unit="pair"):
        base_a = df.loc[idx_a]
        base_b = df.loc[idx_b]
        pair_type = classify_pair(df, idx_a, idx_b)
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
        examples: list[tuple[list[float], float, dict[str, float]]] = []
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
                bridge_depth = features[10]
                one_sided_penalty = features[-1]
                if pair_stats["sim_min"] >= 0.12 and bridge_depth > 0:
                    relevance = 0.10 + 0.05 * pair_stats["sim_min"] + 0.04 * bridge_depth
                    if pair_type != "similar_taste":
                        relevance += 0.05
                elif pair_stats["sim_gap"] >= 0.28 or one_sided_penalty >= 1.5:
                    relevance = -0.12 - 0.05 * min(pair_stats["sim_gap"], 1.0)
                else:
                    relevance = 0.01 * pair_stats["sim_mean"]
            examples.append((features, float(relevance), pair_stats))

        if len(examples) < 2:
            continue

        positives = [
            (features, relevance)
            for features, relevance, pair_stats in examples
            if relevance >= 0.30 and pair_stats["sim_min"] >= 0.12
        ]
        negatives = [
            (features, relevance)
            for features, relevance, pair_stats in examples
            if relevance <= 0.02 or pair_stats["sim_gap"] >= 0.32
        ]
        if not positives or not negatives:
            ranked = sorted(examples, key=lambda item: item[1], reverse=True)
            cut = max(1, len(ranked) // 5)
            positives = [(features, relevance) for features, relevance, _ in ranked[:cut]]
            negatives = [(features, relevance) for features, relevance, _ in ranked[-cut:]]

        pair_multiplier = {
            "similar_taste": 1,
            "mixed_taste": 2,
            "far_apart": 2,
        }[pair_type]
        pos_subset = positives[: min(6 * pair_multiplier, len(positives))]
        neg_subset = negatives[: min(6, len(negatives))]
        for pos_features, _ in pos_subset:
            for neg_features, _ in neg_subset:
                diff = [p - n for p, n in zip(pos_features, neg_features)]
                inv_diff = [-value for value in diff]
                X.append(diff)
                y.append(1)
                X.append(inv_diff)
                y.append(0)

    if compromise_labels_path:
        _append_human_label_examples(df, model, Path(compromise_labels_path), X, y)

    if not X:
        raise RuntimeError("No reranker examples were generated from proxy or human compromise labels.")

    reranker = PairwiseLinearRanker()
    reranker.fit(X, y)
    return reranker


def _append_human_label_examples(
    df: pd.DataFrame,
    model,
    labels_path: Path,
    X: list[list[float]],
    y: list[int],
) -> None:
    if not labels_path.exists():
        raise FileNotFoundError(f"Compromise labels CSV not found: {labels_path}")

    labeled = pd.read_csv(labels_path).fillna("")
    required_columns = {"movie_id_a", "movie_id_b", "candidate_movie_id", "label"}
    missing = required_columns - set(labeled.columns)
    if missing:
        raise ValueError(f"Compromise labels CSV is missing required columns: {sorted(missing)}")

    labeled["movie_id_a"] = labeled["movie_id_a"].astype(str)
    labeled["movie_id_b"] = labeled["movie_id_b"].astype(str)
    labeled["candidate_movie_id"] = labeled["candidate_movie_id"].astype(str)
    labeled["label"] = labeled["label"].astype(str).str.strip()
    if "label_confidence" in labeled.columns:
        labeled["label_confidence"] = labeled["label_confidence"].astype(str).str.strip().str.lower()
    else:
        labeled["label_confidence"] = ""
    labeled = labeled[labeled["label"].isin(LABEL_STRENGTH)].copy()
    if labeled.empty:
        return

    id_to_index = {
        str(movie_id): int(idx)
        for idx, movie_id in enumerate(df["movie_id"].astype(str).tolist())
    }
    label_groups = labeled.groupby(["movie_id_a", "movie_id_b"], sort=False)

    for (movie_id_a, movie_id_b), group in label_groups:
        idx_a = id_to_index.get(movie_id_a)
        idx_b = id_to_index.get(movie_id_b)
        if idx_a is None or idx_b is None:
            continue

        base_a = df.loc[idx_a]
        base_b = df.loc[idx_b]
        score_bundle = model.pair_score_bundle(idx_a, idx_b)
        pair_scores, _ = model.two_seed_candidate_scores(
            idx_a,
            idx_b,
            top_pool=180,
            score_bundle=score_bundle,
        )

        examples: list[tuple[list[float], float, int]] = []
        for _, row in group.iterrows():
            cand_idx = id_to_index.get(str(row["candidate_movie_id"]))
            if cand_idx is None:
                continue
            cand = df.loc[cand_idx]
            pair_stats = model.pair_feature_stats(
                idx_a,
                idx_b,
                cand_idx,
                pair_scores=pair_scores,
                sim_a_scores=score_bundle["sim_a"],
                sim_b_scores=score_bundle["sim_b"],
                joint_scores=score_bundle["joint"],
            )
            features = build_features([base_a, base_b], cand, pair_stats)
            label_strength = LABEL_STRENGTH[str(row["label"]).strip()]
            confidence = CONFIDENCE_MULTIPLIER.get(str(row.get("label_confidence", "")).strip().lower(), 1)
            examples.append((features, label_strength, confidence))

        if len(examples) < 2:
            continue

        examples.sort(key=lambda item: item[1], reverse=True)
        for i, (pos_features, pos_strength, pos_conf) in enumerate(examples):
            for neg_features, neg_strength, neg_conf in examples[i + 1 :]:
                if pos_strength <= neg_strength:
                    continue
                repeats = max(pos_conf, neg_conf)
                diff = [p - n for p, n in zip(pos_features, neg_features)]
                inv_diff = [-value for value in diff]
                for _ in range(repeats):
                    X.append(diff)
                    y.append(1)
                    X.append(inv_diff)
                    y.append(0)
