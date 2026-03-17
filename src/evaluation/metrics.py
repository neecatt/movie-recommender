from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    recs = list(recommended)[:k]
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(recs)


def recall_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    recs = list(recommended)[:k]
    if not relevant:
        return 0.0
    hits = sum(1 for r in recs if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    recs = list(recommended)[:k]
    if not recs:
        return 0.0

    def dcg(items: list[str]) -> float:
        score = 0.0
        for idx, item in enumerate(items, start=1):
            if item in relevant:
                score += 1.0 / math.log2(idx + 1)
        return score

    ideal = list(relevant)[:k]
    return dcg(recs) / (dcg(ideal) + 1e-9)


def ndcg_at_k_weighted(recommended: Iterable[str], gains: dict[str, float], k: int) -> float:
    recs = list(recommended)[:k]
    if not recs:
        return 0.0

    def dcg(items: list[str]) -> float:
        score = 0.0
        for idx, item in enumerate(items, start=1):
            gain = gains.get(item, 0.0)
            if gain > 0:
                score += gain / math.log2(idx + 1)
        return score

    ideal = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    ideal_items = [name for name, _ in ideal][:k]
    return dcg(recs) / (dcg(ideal_items) + 1e-9)
