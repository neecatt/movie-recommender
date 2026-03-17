from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def parse_tmdb_recommendations(value: object) -> set[str]:
    if pd.isna(value) or value in {"", "None"}:
        return set()
    return {part.strip() for part in str(value).split("-") if part.strip()}


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


def build_pair_queries(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
    candidate_pool_size: int = 40,
    min_shared_recommendations: int = 1,
) -> list[tuple[int, int]]:
    recommendation_sets, _ = build_tmdb_recommendation_sets(df)
    primary_genres = (
        df.get("genre", "")
        .fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .tolist()
    )

    by_genre: dict[str, list[int]] = defaultdict(list)
    eligible = []
    for idx, recs in enumerate(recommendation_sets):
        if recs:
            genre = primary_genres[idx]
            if genre:
                by_genre[genre].append(idx)
                eligible.append(idx)

    if not eligible:
        return []

    rng = np.random.default_rng(random_state)
    rng.shuffle(eligible)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for idx_a in eligible:
        genre = primary_genres[idx_a]
        bucket = [idx for idx in by_genre.get(genre, []) if idx != idx_a]
        if not bucket:
            continue

        if len(bucket) > candidate_pool_size:
            candidate_indices = rng.choice(bucket, size=candidate_pool_size, replace=False).tolist()
        else:
            candidate_indices = bucket

        best_idx: int | None = None
        best_score = (-1, -1.0)
        recs_a = recommendation_sets[idx_a]
        genres_a = {g.strip() for g in str(df.iloc[idx_a].get("genre", "")).split(",") if g.strip()}

        for idx_b in candidate_indices:
            recs_b = recommendation_sets[idx_b]
            shared_recs = len(recs_a & recs_b)
            if shared_recs < min_shared_recommendations:
                continue
            genres_b = {g.strip() for g in str(df.iloc[idx_b].get("genre", "")).split(",") if g.strip()}
            score = (shared_recs, float(len(genres_a & genres_b)))
            if score > best_score:
                best_score = score
                best_idx = idx_b

        if best_idx is None:
            continue

        pair = tuple(sorted((idx_a, best_idx)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= sample_size:
            break

    return pairs


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

    seed_genres = []
    for idx in (idx_a, idx_b):
        seed_genres.append(
            {g.strip() for g in str(df.iloc[idx].get("genre", "")).split(",") if g.strip()}
        )

    gains: dict[str, float] = {}
    for rec_id in recs_a | recs_b:
        idx = id_to_index.get(rec_id)
        if idx is None:
            continue
        movie_name = str(df.iloc[idx].get("movie_name", "")).strip()
        if not movie_name:
            continue

        candidate_genres = {
            g.strip() for g in str(df.iloc[idx].get("genre", "")).split(",") if g.strip()
        }
        shared_with_a = bool(candidate_genres & seed_genres[0])
        shared_with_b = bool(candidate_genres & seed_genres[1])

        gain = 1.0
        if rec_id in recs_a and rec_id in recs_b:
            gain += 2.0
        if shared_with_a and shared_with_b:
            gain += 0.5
        elif shared_with_a or shared_with_b:
            gain += 0.25

        gains[movie_name] = max(gain, gains.get(movie_name, 0.0))

    return gains
