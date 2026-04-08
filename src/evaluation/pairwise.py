from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def parse_tmdb_recommendations(value: object) -> set[str]:
    if pd.isna(value) or value in {"", "None"}:
        return set()
    return {part.strip() for part in str(value).split("-") if part.strip()}


def split_genres(value: object) -> set[str]:
    return {part.strip() for part in str(value).split(",") if part.strip()}


def build_tmdb_recommendation_sets(df: pd.DataFrame) -> tuple[list[set[str]], dict[str, int]]:
    if "movie_id" not in df.columns:
        raise ValueError("Expected a movie_id column for TMDB recommendation lookup.")

    id_to_index = {
        str(movie_id): int(idx)
        for idx, movie_id in enumerate(df["movie_id"].astype(str).tolist())
    }

    recommendation_sets: list[set[str]] = []
    for movie_id, raw in zip(df["movie_id"].astype(str), df.get("tmdb_recommendations", "").fillna("")):
        rec_ids = parse_tmdb_recommendations(raw)
        rec_ids.discard(movie_id)
        recommendation_sets.append({rec_id for rec_id in rec_ids if rec_id in id_to_index})

    return recommendation_sets, id_to_index


def classify_pair(df: pd.DataFrame, idx_a: int, idx_b: int) -> str:
    genres_a = split_genres(df.iloc[idx_a].get("genre", ""))
    genres_b = split_genres(df.iloc[idx_b].get("genre", ""))
    primary_a = str(df.iloc[idx_a].get("genre", "")).split(",")[0].strip()
    primary_b = str(df.iloc[idx_b].get("genre", "")).split(",")[0].strip()
    overlap = genres_a & genres_b
    if primary_a and primary_a == primary_b:
        return "similar_taste"
    if overlap:
        return "mixed_taste"
    return "far_apart"


def build_pair_queries(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
    candidate_pool_size: int = 30,
    min_shared_recommendations: int = 1,
) -> list[tuple[int, int]]:
    recommendation_sets, _ = build_tmdb_recommendation_sets(df)
    eligible = [idx for idx, recs in enumerate(recommendation_sets) if recs]
    if not eligible:
        return []

    rng = np.random.default_rng(random_state)
    pairs_by_slice: dict[str, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()

    for idx_a in eligible:
        pool = [idx for idx in eligible if idx != idx_a]
        if len(pool) > candidate_pool_size:
            candidate_indices = rng.choice(pool, size=candidate_pool_size, replace=False).tolist()
        else:
            candidate_indices = pool

        recs_a = recommendation_sets[idx_a]
        genres_a = split_genres(df.iloc[idx_a].get("genre", ""))
        for idx_b in candidate_indices:
            pair = tuple(sorted((idx_a, idx_b)))
            if pair in seen:
                continue
            recs_b = recommendation_sets[idx_b]
            shared_recs = len(recs_a & recs_b)
            if shared_recs < min_shared_recommendations:
                continue
            genres_b = split_genres(df.iloc[idx_b].get("genre", ""))
            overlap = len(genres_a & genres_b)
            pair_type = classify_pair(df, idx_a, idx_b)
            score = (shared_recs, overlap)
            pairs_by_slice[pair_type].append((idx_a, idx_b, score))
            seen.add(pair)

    ordered_pairs: list[tuple[int, int]] = []
    per_slice_quota = max(sample_size // 3, 1)
    for slice_name in ("similar_taste", "mixed_taste", "far_apart"):
        ranked = sorted(pairs_by_slice.get(slice_name, []), key=lambda item: item[2], reverse=True)
        ordered_pairs.extend((idx_a, idx_b) for idx_a, idx_b, _ in ranked[:per_slice_quota])

    if len(ordered_pairs) < sample_size:
        leftovers: list[tuple[int, int, tuple[int, int]]] = []
        for slice_name in ("similar_taste", "mixed_taste", "far_apart"):
            ranked = sorted(pairs_by_slice.get(slice_name, []), key=lambda item: item[2], reverse=True)
            leftovers.extend(ranked[per_slice_quota:])
        leftovers = sorted(leftovers, key=lambda item: item[2], reverse=True)
        ordered_pairs.extend((idx_a, idx_b) for idx_a, idx_b, _ in leftovers[: sample_size - len(ordered_pairs)])

    return ordered_pairs[:sample_size]


def build_pair_relevance_gains(
    idx_a: int,
    idx_b: int,
    df: pd.DataFrame,
    recommendation_sets: list[set[str]],
    id_to_index: dict[str, int],
) -> dict[str, float]:
    recs_a = recommendation_sets[idx_a]
    recs_b = recommendation_sets[idx_b]
    if not recs_a or not recs_b:
        return {}

    seed_rows = [df.iloc[idx_a], df.iloc[idx_b]]
    seed_genres = [split_genres(row.get("genre", "")) for row in seed_rows]
    seed_rating_mean = float(np.mean([float(row.get("rating", 0) or 0) for row in seed_rows]))
    seed_year_mean = float(np.mean([float(row.get("year", 0) or 0) for row in seed_rows if float(row.get("year", 0) or 0) > 0] or [0]))

    gains: dict[str, float] = {}
    for rec_id in recs_a | recs_b:
        idx = id_to_index.get(rec_id)
        if idx is None:
            continue

        row = df.iloc[idx]
        movie_id = str(row.get("movie_id", "")).strip()
        if not movie_id:
            continue

        candidate_genres = split_genres(row.get("genre", ""))
        shared_with_a = bool(candidate_genres & seed_genres[0])
        shared_with_b = bool(candidate_genres & seed_genres[1])
        rating = float(row.get("rating", 0) or 0)
        year = float(row.get("year", 0) or 0)

        gain = 0.0
        if rec_id in recs_a and rec_id in recs_b:
            gain += 2.5
        elif rec_id in recs_a or rec_id in recs_b:
            gain += 0.8

        if shared_with_a and shared_with_b:
            gain += 1.0
        elif shared_with_a or shared_with_b:
            gain += 0.2

        if rating >= max(seed_rating_mean - 1.0, 6.0):
            gain += 0.3
        if year > 0 and seed_year_mean > 0:
            gain += max(0.0, 0.2 - min(abs(seed_year_mean - year) / 100.0, 0.2))

        gains[movie_id] = max(gain, gains.get(movie_id, 0.0))

    return gains
