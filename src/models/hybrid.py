from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.features.graph_features import (
    build_cooccurrence_edges,
    build_cooccurrence_map,
    cooccurrence_similarity,
)
from src.features.text_features import build_text_features
from src.models.reranker import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_features

ARTIFACT_VERSION = "2.0"


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a saved artifact is incompatible with the current code."""


@dataclass
class Recommendation:
    movie_id: str
    movie_name: str
    year: float | None
    rating: float | None
    genre: str | None
    score: float
    explanation: dict[str, list[str]]
    debug_scores: dict[str, float] | None = None


class HybridRecommender:
    """Hybrid recommender for shared movie picks."""

    def __init__(
        self,
        content_weight: float = 0.25,
        embedding_weight: float = 0.30,
        popularity_weight: float = 0.10,
        bm25_weight: float = 0.05,
        genre_weight: float = 0.30,
        min_votes: int = 500,
        use_embeddings: bool = True,
        embedding_model: str = "all-mpnet-base-v2",
        embedding_cache_path: str | None = None,
        embedding_device: str | None = None,
        use_bm25: bool = True,
        use_faiss: bool = True,
        faiss_top_k: int = 3000,
    ) -> None:
        self.content_weight = content_weight
        self.embedding_weight = embedding_weight
        self.popularity_weight = popularity_weight
        self.bm25_weight = bm25_weight
        self.genre_weight = genre_weight
        self.min_votes = min_votes
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.embedding_cache_path = embedding_cache_path
        self.embedding_device = embedding_device
        self.use_bm25 = use_bm25
        self.use_faiss = use_faiss
        self.faiss_top_k = faiss_top_k
        self._df: pd.DataFrame | None = None
        self._vectorizer: Any | None = None
        self._tfidf_matrix: Any | None = None
        self._embedding_matrix: Any | None = None
        self._bm25: Any | None = None
        self._tokenized: list[list[str]] | None = None
        self._id_to_index: pd.Series | None = None
        self._title_to_indices: pd.Series | None = None
        self._genre_co_map: dict[tuple[str, str], float] = {}
        self._genre_matrix: np.ndarray | None = None
        self._reranker: Any | None = None
        self._faiss_index: Any | None = None
        self._faiss_vectors: Any | None = None
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = (
            self.content_weight
            + self.embedding_weight
            + self.popularity_weight
            + self.bm25_weight
            + self.genre_weight
        )
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        self.content_weight /= total
        self.embedding_weight /= total
        self.popularity_weight /= total
        self.bm25_weight /= total
        self.genre_weight /= total

    def fit(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True).copy()
        if "movie_id" in self._df.columns:
            self._df["movie_id"] = self._df["movie_id"].astype(str)
        if "movie_name" in self._df.columns:
            self._df["movie_name"] = self._df["movie_name"].astype(str)
        (
            self._vectorizer,
            self._tfidf_matrix,
            self._embedding_matrix,
            self._tokenized,
        ) = build_text_features(
            self._df,
            use_embeddings=self.use_embeddings,
            embedding_model=self.embedding_model,
            cache_path=Path(self.embedding_cache_path) if self.embedding_cache_path else None,
            embedding_device=self.embedding_device,
        )
        if self.use_bm25 and self._tokenized:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized)
        self._build_faiss_index()
        self._build_genre_matrix()
        if "genre" in self._df.columns:
            edges = build_cooccurrence_edges(self._df, "genre")
            self._genre_co_map = build_cooccurrence_map(edges)
        self._id_to_index = self._df.reset_index().set_index("movie_id")["index"]
        self._title_to_indices = self._df.reset_index().groupby("movie_name")["index"].agg(list)

    @staticmethod
    def supported_modes() -> tuple[str, ...]:
        return ("single", "pair")

    @staticmethod
    def expected_reranker_feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES

    @classmethod
    def validate_artifacts(cls, artifacts: dict[str, Any]) -> None:
        version = artifacts.get("artifact_version")
        if version != ARTIFACT_VERSION:
            raise ArtifactCompatibilityError(
                f"Unsupported artifact version: {version!r}. Expected {ARTIFACT_VERSION}."
            )

        schema_version = artifacts.get("feature_schema_version")
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ArtifactCompatibilityError(
                f"Unsupported feature schema: {schema_version!r}. Expected {FEATURE_SCHEMA_VERSION}."
            )

        feature_names = tuple(artifacts.get("reranker_feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ArtifactCompatibilityError(
                "Saved reranker feature names do not match the current code schema."
            )

        reranker = artifacts.get("reranker")
        expected_count = len(FEATURE_NAMES)
        saved_count = int(artifacts.get("reranker_n_features", expected_count))
        if saved_count != expected_count:
            raise ArtifactCompatibilityError(
                f"Saved reranker expects {saved_count} features but the code expects {expected_count}."
            )
        if reranker is not None and getattr(reranker, "n_features_in_", expected_count) != expected_count:
            raise ArtifactCompatibilityError(
                "Saved reranker model is incompatible with the current feature schema."
            )

    def export_artifacts(self) -> dict[str, Any]:
        if (
            self._df is None
            or self._tfidf_matrix is None
            or self._vectorizer is None
            or self._id_to_index is None
        ):
            raise RuntimeError("Call fit() before export_artifacts().")
        return {
            "artifact_version": ARTIFACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_timestamp": datetime.now(UTC).isoformat(),
            "supported_modes": list(self.supported_modes()),
            "reranker_feature_names": list(FEATURE_NAMES),
            "reranker_n_features": len(FEATURE_NAMES),
            "df": self._df,
            "vectorizer": self._vectorizer,
            "tfidf_matrix": self._tfidf_matrix,
            "embedding_matrix": self._embedding_matrix,
            "bm25": self._bm25,
            "tokenized": self._tokenized,
            "id_to_index": self._id_to_index,
            "title_to_indices": self._title_to_indices,
            "genre_matrix": self._genre_matrix,
            "content_weight": self.content_weight,
            "embedding_weight": self.embedding_weight,
            "popularity_weight": self.popularity_weight,
            "bm25_weight": self.bm25_weight,
            "genre_weight": self.genre_weight,
            "min_votes": self.min_votes,
            "use_embeddings": self.use_embeddings,
            "embedding_model": self.embedding_model,
            "embedding_cache_path": self.embedding_cache_path,
            "embedding_device": self.embedding_device,
            "use_bm25": self.use_bm25,
            "reranker": self._reranker,
            "use_faiss": self.use_faiss,
            "faiss_top_k": self.faiss_top_k,
        }

    @classmethod
    def from_artifacts(cls, artifacts: dict[str, Any], validate: bool = True) -> "HybridRecommender":
        if validate:
            cls.validate_artifacts(artifacts)
        model = cls(
            content_weight=float(artifacts.get("content_weight", 0.25)),
            embedding_weight=float(artifacts.get("embedding_weight", 0.30)),
            popularity_weight=float(artifacts.get("popularity_weight", 0.10)),
            bm25_weight=float(artifacts.get("bm25_weight", 0.05)),
            genre_weight=float(artifacts.get("genre_weight", 0.30)),
            min_votes=int(artifacts.get("min_votes", 500)),
            use_embeddings=bool(artifacts.get("use_embeddings", True)),
            embedding_model=str(artifacts.get("embedding_model", "all-mpnet-base-v2")),
            embedding_cache_path=artifacts.get("embedding_cache_path"),
            embedding_device=artifacts.get("embedding_device"),
            use_bm25=bool(artifacts.get("use_bm25", False)),
            use_faiss=bool(artifacts.get("use_faiss", True)),
            faiss_top_k=int(artifacts.get("faiss_top_k", 3000)),
        )
        model._df = artifacts["df"].copy()
        if "movie_id" in model._df.columns:
            model._df["movie_id"] = model._df["movie_id"].astype(str)
        if "movie_name" in model._df.columns:
            model._df["movie_name"] = model._df["movie_name"].astype(str)
        model._vectorizer = artifacts["vectorizer"]
        model._tfidf_matrix = artifacts["tfidf_matrix"]
        model._embedding_matrix = artifacts.get("embedding_matrix")
        model._bm25 = artifacts.get("bm25")
        model._tokenized = artifacts.get("tokenized")
        model._id_to_index = artifacts.get("id_to_index")
        if model._id_to_index is None:
            model._id_to_index = model._df.reset_index().set_index("movie_id")["index"]
        model._title_to_indices = artifacts.get("title_to_indices")
        if model._title_to_indices is None:
            model._title_to_indices = model._df.reset_index().groupby("movie_name")["index"].agg(list)
        model._genre_matrix = artifacts.get("genre_matrix")
        if model._genre_matrix is None:
            model._build_genre_matrix()
        model._build_faiss_index()
        model._reranker = artifacts.get("reranker")
        if "genre" in model._df.columns:
            edges = build_cooccurrence_edges(model._df, "genre")
            model._genre_co_map = build_cooccurrence_map(edges)
        return model

    def _build_genre_matrix(self) -> None:
        if self._df is None or "genre" not in self._df.columns:
            self._genre_matrix = None
            return
        all_genres: set[str] = set()
        genre_lists: list[list[str]] = []
        for val in self._df["genre"].fillna(""):
            genres = [g.strip() for g in str(val).split(",") if g.strip()]
            genre_lists.append(genres)
            all_genres.update(genres)
        sorted_genres = sorted(all_genres)
        genre_to_idx = {genre: i for i, genre in enumerate(sorted_genres)}
        matrix = np.zeros((len(self._df), len(sorted_genres)), dtype=np.float32)
        for i, genres in enumerate(genre_lists):
            for genre in genres:
                matrix[i, genre_to_idx[genre]] = 1.0
        self._genre_matrix = matrix

    def _build_faiss_index(self) -> None:
        if not self.use_faiss or self._embedding_matrix is None:
            self._faiss_index = None
            self._faiss_vectors = None
            return
        try:
            import faiss
        except ImportError:
            self._faiss_index = None
            self._faiss_vectors = None
            return
        vectors = self._embedding_matrix.astype("float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss_index = index
        self._faiss_vectors = vectors

    def set_reranker(self, reranker: Any) -> None:
        self._reranker = reranker

    def search_movies(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if self._df is None or not query.strip():
            return []
        working = self._df.copy()
        q = query.strip().lower()
        names = working["movie_name"].fillna("").astype(str)
        lower_names = names.str.lower()
        mask = lower_names.str.contains(q, regex=False)
        if not mask.any():
            return []
        matched = working.loc[mask].copy()
        matched["_exact"] = (matched["movie_name"].str.lower() == q).astype(int)
        matched["_starts"] = matched["movie_name"].str.lower().str.startswith(q).astype(int)
        matched = matched.sort_values(
            by=["_exact", "_starts", "rating", "votes"],
            ascending=[False, False, False, False],
        ).head(limit)
        return [
            {
                "movie_id": str(row.get("movie_id", "")),
                "movie_name": row.get("movie_name", ""),
                "year": None if pd.isna(row.get("year")) else int(row.get("year")),
                "genre": row.get("genre"),
                "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
            }
            for _, row in matched.iterrows()
        ]

    def _resolve_reference(self, movie_ref: str) -> int:
        if self._df is None or self._id_to_index is None or self._title_to_indices is None:
            raise RuntimeError("Call fit() before recommend().")
        movie_ref = str(movie_ref)
        if movie_ref in self._id_to_index:
            return int(self._id_to_index[movie_ref])
        if movie_ref in self._title_to_indices:
            indices = self._title_to_indices[movie_ref]
            if len(indices) == 1:
                return int(indices[0])
            raise ValueError(f"Ambiguous title: {movie_ref}. Use movie_id instead.")
        raise ValueError(f"Movie not found: {movie_ref}")

    def _movie_payload(self, row: pd.Series) -> dict[str, Any]:
        return {
            "movie_id": str(row.get("movie_id", "")),
            "movie_name": row.get("movie_name", ""),
            "year": None if pd.isna(row.get("year")) else int(row.get("year")),
            "rating": None if pd.isna(row.get("rating")) else float(row.get("rating")),
            "genre": row.get("genre"),
        }

    def _genre_similarity(self, idx_list: list[int]) -> np.ndarray:
        if self._genre_matrix is None or self._df is None:
            return np.zeros(len(self._df) if self._df is not None else 0)
        sims = []
        for idx in idx_list:
            seed = self._genre_matrix[idx]
            intersection = np.minimum(self._genre_matrix, seed).sum(axis=1)
            union = np.maximum(self._genre_matrix, seed).sum(axis=1)
            sims.append(intersection / (union + 1e-9))
        return self._aggregate_multi_seed(sims)

    @staticmethod
    def _aggregate_multi_seed(sim_list: list[np.ndarray]) -> np.ndarray:
        if len(sim_list) == 1:
            return sim_list[0]
        stacked = np.vstack(sim_list)
        avg = np.mean(stacked, axis=0)
        minimum = np.min(stacked, axis=0)
        return 0.35 * avg + 0.65 * minimum

    def _popularity_score(self) -> np.ndarray:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        votes = self._df.get("votes", pd.Series(0, index=self._df.index)).fillna(0).to_numpy()
        ratings = self._df.get("rating", pd.Series(0, index=self._df.index)).fillna(0).to_numpy()
        return ratings * np.log1p(votes)

    def _candidate_mask(self) -> np.ndarray:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        votes = self._df.get("votes", pd.Series(0, index=self._df.index)).fillna(0)
        return (votes >= self.min_votes).to_numpy()

    def _faiss_candidates(self, idx_list: list[int]) -> set[int]:
        if self._faiss_index is None or self._faiss_vectors is None:
            return set()
        query = self._faiss_vectors[idx_list]
        _, indices = self._faiss_index.search(query, self.faiss_top_k)
        return set(np.unique(indices).tolist())

    @staticmethod
    def _split_items(value: object) -> set[str]:
        return {item.strip() for item in str(value).split(",") if item.strip()}

    def _quality_bonus(self, cand_row: pd.Series) -> float:
        bayes = float(cand_row.get("bayesian_rating", 0) or 0) / 10.0
        recency = float(cand_row.get("recency_score", 0) or 0)
        return 0.7 * bayes + 0.3 * recency

    def _paired_overlap_bonus(
        self,
        base_rows: list[pd.Series],
        cand_row: pd.Series,
    ) -> float:
        bonus = 0.0
        cand_genres = self._split_items(cand_row.get("genre", ""))
        per_seed_overlap: list[int] = []
        for base_row in base_rows:
            base_genres = self._split_items(base_row.get("genre", ""))
            overlap = cand_genres & base_genres
            per_seed_overlap.append(len(overlap))
            bonus += 0.12 * len(overlap)
            if self._genre_co_map and cand_genres and base_genres:
                bonus += 0.005 * cooccurrence_similarity(base_genres, cand_genres, self._genre_co_map)
        if per_seed_overlap:
            if min(per_seed_overlap) > 0:
                bonus += 0.18 + 0.05 * min(per_seed_overlap)
            else:
                bonus -= 0.08 * (max(per_seed_overlap) - min(per_seed_overlap))
        return bonus + 0.2 * self._quality_bonus(cand_row)

    def _pair_bridge_scores(self, idx_a: int, idx_b: int) -> np.ndarray:
        if self._genre_matrix is None or self._df is None:
            return np.zeros(len(self._df) if self._df is not None else 0, dtype=np.float32)
        overlap_a = np.minimum(self._genre_matrix, self._genre_matrix[idx_a]).sum(axis=1)
        overlap_b = np.minimum(self._genre_matrix, self._genre_matrix[idx_b]).sum(axis=1)
        bridge_binary = ((overlap_a > 0) & (overlap_b > 0)).astype(np.float32)
        bridge_depth = np.minimum(overlap_a, overlap_b).astype(np.float32)
        if bridge_depth.max() > bridge_depth.min():
            bridge_depth = (bridge_depth - bridge_depth.min()) / (bridge_depth.max() - bridge_depth.min() + 1e-9)
        else:
            bridge_depth = np.zeros_like(bridge_depth)
        return 0.65 * bridge_binary + 0.35 * bridge_depth

    def _combined_scores(self, idx_list: list[int]) -> np.ndarray:
        if self._df is None or self._tfidf_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        tfidf_sims = [linear_kernel(self._tfidf_matrix[idx], self._tfidf_matrix).flatten() for idx in idx_list]
        tfidf_sim = self._aggregate_multi_seed(tfidf_sims)

        embed_sim = np.zeros_like(tfidf_sim)
        if self._embedding_matrix is not None:
            embed_sims = [
                cosine_similarity(self._embedding_matrix[idx].reshape(1, -1), self._embedding_matrix).flatten()
                for idx in idx_list
            ]
            embed_sim = self._aggregate_multi_seed(embed_sims)

        genre_sim = self._genre_similarity(idx_list)
        pop_scores = self._popularity_score()
        bm25_scores = np.zeros_like(tfidf_sim)
        if self._bm25 is not None and self._tokenized is not None:
            bm25_sims = [self._bm25.get_scores(self._tokenized[idx]) for idx in idx_list]
            bm25_scores = self._aggregate_multi_seed(bm25_sims)

        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-9)

        tfidf_sim = _minmax(tfidf_sim)
        if embed_sim.max() > embed_sim.min():
            embed_sim = _minmax(embed_sim)
        genre_sim = _minmax(genre_sim)
        pop_scores = _minmax(pop_scores)
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = _minmax(bm25_scores)

        return (
            self.content_weight * tfidf_sim
            + self.embedding_weight * embed_sim
            + self.genre_weight * genre_sim
            + self.popularity_weight * pop_scores
            + self.bm25_weight * bm25_scores
        )

    def pair_feature_stats(
        self,
        idx_a: int,
        idx_b: int,
        cand_idx: int,
        pair_scores: np.ndarray | None = None,
        sim_a_scores: np.ndarray | None = None,
        sim_b_scores: np.ndarray | None = None,
        joint_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        if sim_a_scores is None:
            sim_a_scores = self._combined_scores([idx_a])
        if sim_b_scores is None:
            sim_b_scores = self._combined_scores([idx_b])
        if joint_scores is None:
            joint_scores = self._combined_scores([idx_a, idx_b])

        sim_a = float(sim_a_scores[cand_idx])
        sim_b = float(sim_b_scores[cand_idx])
        joint_score = float(joint_scores[cand_idx])
        pair_score = (
            float(pair_scores[cand_idx])
            if pair_scores is not None
            else 0.6 * min(sim_a, sim_b) + 0.3 * joint_score + 0.1 * ((sim_a + sim_b) / 2.0) - 0.2 * abs(sim_a - sim_b)
        )
        return {
            "sim_a": sim_a,
            "sim_b": sim_b,
            "sim_min": min(sim_a, sim_b),
            "sim_mean": (sim_a + sim_b) / 2.0,
            "sim_gap": abs(sim_a - sim_b),
            "joint_score": joint_score,
            "pair_score": pair_score,
        }

    def pair_score_bundle(self, idx_a: int, idx_b: int) -> dict[str, np.ndarray]:
        sim_a = self._combined_scores([idx_a])
        sim_b = self._combined_scores([idx_b])
        joint = self._combined_scores([idx_a, idx_b])
        bridge = self._pair_bridge_scores(idx_a, idx_b)
        pair_scores = (
            0.35 * np.minimum(sim_a, sim_b)
            + 0.20 * joint
            + 0.15 * ((sim_a + sim_b) / 2.0)
            + 0.30 * bridge
            - 0.10 * np.abs(sim_a - sim_b)
        )
        return {
            "sim_a": sim_a,
            "sim_b": sim_b,
            "joint": joint,
            "bridge": bridge,
            "pair_scores": pair_scores,
        }

    def two_seed_candidate_scores(
        self,
        idx_a: int,
        idx_b: int,
        top_pool: int = 300,
        score_bundle: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")

        bundle = score_bundle or self.pair_score_bundle(idx_a, idx_b)
        sim_a = bundle["sim_a"]
        sim_b = bundle["sim_b"]
        joint = bundle["joint"]
        pair_scores = bundle["pair_scores"]

        mask = self._candidate_mask()
        base_exclusions = {idx_a, idx_b}
        faiss_candidates = self._faiss_candidates([idx_a, idx_b])

        candidate_pool: set[int] = set()
        per_seed_pool = max(top_pool, 120)
        for scores in (sim_a, sim_b, joint, bundle["bridge"], pair_scores):
            ranked = scores.argsort()[::-1]
            count = 0
            for idx in ranked:
                if idx in base_exclusions or not mask[idx]:
                    continue
                if faiss_candidates and idx not in faiss_candidates:
                    continue
                candidate_pool.add(int(idx))
                count += 1
                if count >= per_seed_pool:
                    break

        ordered_candidates = sorted(candidate_pool, key=lambda idx: float(pair_scores[idx]), reverse=True)
        return pair_scores, ordered_candidates

    def set_weights(
        self,
        content_weight: float,
        embedding_weight: float,
        popularity_weight: float,
        bm25_weight: float,
        genre_weight: float,
        min_votes: int,
    ) -> None:
        self.content_weight = content_weight
        self.embedding_weight = embedding_weight
        self.popularity_weight = popularity_weight
        self.bm25_weight = bm25_weight
        self.genre_weight = genre_weight
        self.min_votes = min_votes
        self._normalize_weights()

    def recommend(self, movie_ref: str, top_n: int = 10) -> list[Recommendation]:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        idx = self._resolve_reference(movie_ref)
        combined = self._combined_scores([idx])
        mask = self._candidate_mask()
        faiss_candidates = self._faiss_candidates([idx])
        candidates = [
            i
            for i in combined.argsort()[::-1]
            if i != idx and mask[i] and (not faiss_candidates or i in faiss_candidates)
        ][: max(top_n * 5, 50)]

        base_row = self._df.iloc[idx]
        reranked = []
        for i in candidates:
            row = self._df.iloc[i]
            final_score = float(combined[i]) + self._paired_overlap_bonus([base_row], row)
            reranked.append((i, final_score))
        reranked = sorted(reranked, key=lambda item: item[1], reverse=True)[:top_n]
        return [
            Recommendation(
                movie_id=str(self._df.iloc[i].get("movie_id", "")),
                movie_name=self._df.iloc[i].get("movie_name", ""),
                year=self._df.iloc[i].get("year"),
                rating=self._df.iloc[i].get("rating"),
                genre=self._df.iloc[i].get("genre"),
                score=score,
                explanation={
                    "works_for_both": [],
                    "leans_to_a": [f"Matches {base_row.get('movie_name', 'the input movie')} through shared themes."],
                    "leans_to_b": [],
                },
            )
            for i, score in reranked
        ]

    def _format_seed_overlap_reason(self, seed_label: str, seed_row: pd.Series, cand_row: pd.Series) -> list[str]:
        reasons: list[str] = []
        shared_genres = self._split_items(seed_row.get("genre", "")) & self._split_items(cand_row.get("genre", ""))
        if shared_genres:
            reasons.append(f"Shares genres with {seed_label}: {', '.join(sorted(shared_genres)[:3])}.")
        shared_keywords = self._split_items(seed_row.get("keywords", "")) & self._split_items(cand_row.get("keywords", ""))
        if shared_keywords:
            reasons.append(f"Connects with {seed_label} themes like {', '.join(sorted(shared_keywords)[:3])}.")
        shared_cast = self._split_items(seed_row.get("star", "")) & self._split_items(cand_row.get("star", ""))
        if shared_cast:
            reasons.append(f"Has familiar cast for {seed_label}: {', '.join(sorted(shared_cast)[:3])}.")
        return reasons

    @staticmethod
    def _dedupe_reasons(reasons: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                ordered.append(reason)
        return ordered

    def _build_pair_explanation(
        self,
        base_a: pd.Series,
        base_b: pd.Series,
        cand_row: pd.Series,
        pair_stats: dict[str, float],
    ) -> dict[str, list[str]]:
        works_for_both: list[str] = []
        leans_to_a = self._format_seed_overlap_reason("person A", base_a, cand_row)
        leans_to_b = self._format_seed_overlap_reason("person B", base_b, cand_row)

        genres_a = self._split_items(base_a.get("genre", ""))
        genres_b = self._split_items(base_b.get("genre", ""))
        cand_genres = self._split_items(cand_row.get("genre", ""))
        if cand_genres & genres_a and cand_genres & genres_b:
            works_for_both.append("Bridges both picks through overlapping genres.")
        elif cand_genres and (cand_genres & genres_a or cand_genres & genres_b):
            works_for_both.append("Acts as a compromise by borrowing elements from each side.")
        if pair_stats["sim_min"] >= 0.45:
            works_for_both.append("Stays reasonably close to both people’s tastes.")
        if pair_stats["sim_gap"] <= 0.12:
            works_for_both.append("Feels balanced instead of favoring only one person.")
        if self._quality_bonus(cand_row) >= 0.55:
            works_for_both.append("Has strong audience quality signals for a safe shared watch.")

        return {
            "works_for_both": self._dedupe_reasons(works_for_both),
            "leans_to_a": self._dedupe_reasons(leans_to_a),
            "leans_to_b": self._dedupe_reasons(leans_to_b),
        }

    def _recommend_from_indices(
        self,
        idx_a: int,
        idx_b: int,
        top_n: int = 5,
        include_debug: bool = False,
    ) -> list[Recommendation]:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        if idx_a == idx_b:
            raise ValueError("Choose two different movies.")

        score_bundle = self.pair_score_bundle(idx_a, idx_b)
        pair_scores, candidates = self.two_seed_candidate_scores(
            idx_a,
            idx_b,
            top_pool=max(top_n * 20, 180),
            score_bundle=score_bundle,
        )
        candidates = candidates[: max(top_n * 10, 80)]

        base_a = self._df.iloc[idx_a]
        base_b = self._df.iloc[idx_b]
        reranked: list[tuple[int, float, dict[str, float]]] = []
        for i in candidates:
            row = self._df.iloc[i]
            pair_stats = self.pair_feature_stats(
                idx_a,
                idx_b,
                i,
                pair_scores=pair_scores,
                sim_a_scores=score_bundle["sim_a"],
                sim_b_scores=score_bundle["sim_b"],
                joint_scores=score_bundle["joint"],
            )
            rule_score = self._paired_overlap_bonus([base_a, base_b], row)
            final_score = float(pair_stats["pair_score"]) + rule_score
            if self._reranker is not None:
                features = build_features([base_a, base_b], row, pair_stats)
                final_score += float(self._reranker.predict([features])[0])
            debug_scores = {
                **pair_stats,
                "rule_score": rule_score,
                "final_score": final_score,
            }
            reranked.append((i, final_score, debug_scores))

        reranked = sorted(reranked, key=lambda item: item[1], reverse=True)
        genre_counts: dict[str, int] = {}
        final_indices: list[tuple[int, float, dict[str, float]]] = []
        for i, score, debug_scores in reranked:
            row = self._df.iloc[i]
            primary_genre = str(row.get("genre", "")).split(",")[0].strip()
            penalty = 0.05 * genre_counts.get(primary_genre, 0)
            adjusted_score = score - penalty
            debug_scores = {**debug_scores, "diversity_penalty": penalty, "adjusted_score": adjusted_score}
            final_indices.append((i, adjusted_score, debug_scores))
            genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1
            if len(final_indices) >= top_n:
                break

        results: list[Recommendation] = []
        for i, adjusted_score, debug_scores in final_indices:
            row = self._df.iloc[i]
            pair_stats = {
                "sim_a": debug_scores["sim_a"],
                "sim_b": debug_scores["sim_b"],
                "sim_min": debug_scores["sim_min"],
                "sim_gap": debug_scores["sim_gap"],
            }
            results.append(
                Recommendation(
                    movie_id=str(row.get("movie_id", "")),
                    movie_name=row.get("movie_name", ""),
                    year=row.get("year"),
                    rating=row.get("rating"),
                    genre=row.get("genre"),
                    score=adjusted_score,
                    explanation=self._build_pair_explanation(base_a, base_b, row, pair_stats),
                    debug_scores=debug_scores if include_debug else None,
                )
            )
        return results

    def recommend_from_two(self, movie_id_a: str, movie_id_b: str, top_n: int = 5) -> list[Recommendation]:
        idx_a = self._resolve_reference(movie_id_a)
        idx_b = self._resolve_reference(movie_id_b)
        return self._recommend_from_indices(idx_a, idx_b, top_n=top_n, include_debug=False)

    def recommend_date_movie(
        self,
        movie_id_a: str,
        movie_id_b: str,
        alternatives_n: int = 4,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        idx_a = self._resolve_reference(movie_id_a)
        idx_b = self._resolve_reference(movie_id_b)
        recs = self._recommend_from_indices(idx_a, idx_b, top_n=alternatives_n + 1, include_debug=include_debug)
        if not recs:
            raise ValueError("No shared recommendation could be generated.")

        base_a = self._df.iloc[idx_a]
        base_b = self._df.iloc[idx_b]
        best_pick = recs[0]
        return {
            "movie_a": self._movie_payload(base_a),
            "movie_b": self._movie_payload(base_b),
            "best_pick": self._recommendation_payload(best_pick, include_debug=include_debug),
            "alternatives": [
                self._recommendation_payload(rec, include_debug=include_debug)
                for rec in recs[1 : alternatives_n + 1]
            ],
            "explanation": best_pick.explanation,
        }

    @staticmethod
    def _recommendation_payload(rec: Recommendation, include_debug: bool = False) -> dict[str, Any]:
        payload = {
            "movie_id": rec.movie_id,
            "movie_name": rec.movie_name,
            "year": None if pd.isna(rec.year) else int(rec.year),
            "rating": None if pd.isna(rec.rating) else float(rec.rating),
            "genre": rec.genre,
            "shared_fit_score": rec.score,
            "explanation": rec.explanation,
        }
        if include_debug and rec.debug_scores is not None:
            payload["debug_scores"] = rec.debug_scores
        return payload
