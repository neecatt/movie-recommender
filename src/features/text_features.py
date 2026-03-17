from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_text_features(
    df: pd.DataFrame,
    use_embeddings: bool = False,
    embedding_model: str = "all-mpnet-base-v2",
    cache_path: Path | None = None,
) -> tuple[TfidfVectorizer, object, object | None, list[list[str]]]:
    """Build TF-IDF features and optional sentence embeddings."""
    working = df.copy()
    for col in ["genre", "director", "star", "description", "keywords"]:
        if col in working.columns:
            working[col] = working[col].fillna("")
        else:
            working[col] = ""

    # TF-IDF: repeat genre/keywords/star to boost their weight relative to descriptions
    tfidf_text = (
        working["genre"] + " " + working["genre"] + " " + working["genre"] + " "
        + working["keywords"] + " " + working["keywords"] + " "
        + working["director"] + " " + working["director"] + " "
        + working["star"] + " " + working["star"] + " "
        + working["description"]
    ).str.lower()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(tfidf_text)

    tokenized = [text.split() for text in tfidf_text.tolist()]

    # Embeddings: use natural-language format (transformers handle this better than repetition)
    embedding_matrix = None
    if use_embeddings:
        embed_text = (
            "Genres: " + working["genre"] + ". "
            + "Keywords: " + working["keywords"] + ". "
            + "Director: " + working["director"] + ". "
            + "Stars: " + working["star"] + ". "
            + working["description"]
        ).str.lower()

        if cache_path and cache_path.exists():
            loaded = np.load(cache_path)
            if loaded.shape[0] == len(embed_text):
                embedding_matrix = loaded
        if embedding_matrix is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("sentence-transformers is required for embeddings") from exc
            device = "mps"
            model = SentenceTransformer(embedding_model, device=device)
            embedding_matrix = model.encode(embed_text.tolist(), show_progress_bar=True)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embedding_matrix)

    return vectorizer, matrix, embedding_matrix, tokenized
