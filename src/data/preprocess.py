from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema


def _nonempty_ratio(series: pd.Series) -> float:
    normalized = series.fillna("").astype(str).str.strip()
    return float((normalized != "").mean())


def _normalize_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.replace("  +", " ", regex=True)
                .str.strip()
            )
    return df


def _parse_runtime(value: object) -> pd.Series | pd.NA:
    if pd.isna(value) or value == "" or value == "None":
        return pd.NA
    text = str(value).replace(" min", "").strip()
    return pd.to_numeric(text, errors="coerce")


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw movie dataset."""
    df = df.copy()

    if "gross(in $)" in df.columns:
        df = df.rename(columns={"gross(in $)": "gross"})

    string_cols = [
        "movie_name", "certificate", "genre", "description",
        "director", "director_id", "star", "star_id", "keywords",
        "original_language", "tmdb_recommendations",
    ]
    df = _normalize_strings(df, [c for c in string_cols if c in df.columns])

    if "runtime" in df.columns and "runtime_min" not in df.columns:
        df["runtime_min"] = df["runtime"].apply(_parse_runtime)
        df = df.drop(columns=["runtime"])

    for col in ["year", "rating", "votes", "gross", "runtime_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in {"year", "rating", "votes", "gross", "runtime_min"}:
                df[col] = df[col].astype(float)

    schema = DataFrameSchema(
        {
            "movie_id": Column(pa.String, required=False),
            "movie_name": Column(pa.String, required=False),
            "year": Column(pa.Float, required=False, nullable=True),
            "genre": Column(pa.String, required=False),
            "rating": Column(pa.Float, required=False, nullable=True),
            "description": Column(pa.String, required=False),
            "star": Column(pa.String, required=False),
            "votes": Column(pa.Float, required=False, nullable=True),
            "gross": Column(pa.Float, required=False, nullable=True),
            "runtime_min": Column(pa.Float, required=False, nullable=True),
        },
        strict=False,
        coerce=False,
    )
    df = schema.validate(df, lazy=True)

    if "rating" in df.columns:
        df = df.dropna(subset=["rating"]).reset_index(drop=True)

    if "director" in df.columns:
        df["director_primary"] = df["director"].str.split(",").str[0].str.strip()
    if "star" in df.columns:
        df["star_primary"] = df["star"].str.split(",").str[0].str.strip()

    if "year" in df.columns:
        df["year_decade"] = (df["year"] // 10) * 10
        max_year = df["year"].max()
        df["recency_score"] = (df["year"] / max_year).fillna(0)

    if "rating" in df.columns and "votes" in df.columns:
        global_mean = df["rating"].mean()
        m = df["votes"].median() if df["votes"].notna().any() else 0
        df["bayesian_rating"] = (
            (df["votes"] * df["rating"] + m * global_mean) / (df["votes"] + m)
        )

    if "runtime_min" in df.columns and "genre" in df.columns:
        df["runtime_min"] = df.groupby("genre")["runtime_min"].transform(
            lambda s: s.fillna(s.median())
        )

    if "gross" in df.columns and "year" in df.columns:
        df["gross"] = df.groupby("year")["gross"].transform(lambda s: s.fillna(s.median()))
        df["gross"] = df["gross"].fillna(0)

    required_signal_cols = {
        "movie_id": 0.99,
        "movie_name": 0.99,
        "genre": 0.75,
        "description": 0.55,
        "keywords": 0.20,
        "star": 0.50,
    }
    for col, min_ratio in required_signal_cols.items():
        if col in df.columns:
            ratio = _nonempty_ratio(df[col])
            if ratio < min_ratio:
                raise ValueError(
                    f"Column {col} is too sparse for ranking: coverage={ratio:.3f}, required={min_ratio:.3f}"
                )

    if "movie_id" in df.columns and df["movie_id"].duplicated().any():
        raise ValueError("movie_id must remain unique after preprocessing.")

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed dataset to disk."""
    df.to_csv(output_path, index=False)
