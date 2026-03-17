from __future__ import annotations

import itertools
from collections import Counter

import pandas as pd


def build_cooccurrence_edges(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Build simple co-occurrence edges for a comma-separated column."""
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

    counter: Counter[tuple[str, str]] = Counter()
    for raw in df[column].fillna(""):
        items = [item.strip() for item in str(raw).split(",") if item.strip()]
        for a, b in itertools.combinations(sorted(set(items)), 2):
            counter[(a, b)] += 1

    edges = pd.DataFrame(
        [{"node_a": a, "node_b": b, "weight": w} for (a, b), w in counter.items()]
    )
    if edges.empty:
        return pd.DataFrame(columns=["node_a", "node_b", "weight"])
    return edges.sort_values("weight", ascending=False).reset_index(drop=True)


def build_cooccurrence_map(edges: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (row["node_a"], row["node_b"]): float(row["weight"])
        for _, row in edges.iterrows()
    }


def cooccurrence_similarity(
    base_genres: set[str],
    cand_genres: set[str],
    co_map: dict[tuple[str, str], float],
) -> float:
    score = 0.0
    for a in base_genres:
        for b in cand_genres:
            if a == b:
                score += 1.0
            else:
                score += co_map.get((a, b), 0.0) + co_map.get((b, a), 0.0)
    return score
