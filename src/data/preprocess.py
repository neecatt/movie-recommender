from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema

LITERAL_NULL_TOKENS = {"nan", "none", "null"}


def _nonempty_ratio(series: pd.Series) -> float:
    normalized = series.fillna("").map(lambda value: str(value).strip())
    return float((normalized != "").mean())


def _normalize_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            series = df[col]
            normalized = series.where(series.notna(), "").map(lambda value: str(value))
            df[col] = (
                normalized
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


def _primary_genre(value: object) -> str:
    return str(value or "").split(",")[0].strip()


def _literal_null_ratio(series: pd.Series) -> float:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return float(normalized.isin(LITERAL_NULL_TOKENS).mean())


def _row_has_suspicious_cast_fragment(value: object) -> bool:
    names = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if len(names) < 3:
        return False
    for prev_name, current_name, next_name in zip(names, names[1:], names[2:]):
        if (
            " " in prev_name
            and " " not in current_name
            and " " in next_name
            and current_name.lower() not in {"zendaya", "madonna", "prince"}
        ):
            return True
    return False


def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the raw movie dataset."""
    df = df.copy()

    if "gross(in $)" in df.columns:
        df = df.rename(columns={"gross(in $)": "gross"})

    string_cols = [
        "movie_name",
        "certificate",
        "genre",
        "description",
        "director",
        "director_id",
        "star",
        "star_id",
        "keywords",
        "original_language",
        "tmdb_recommendations",
        "poster_path",
        "production_companies",
    ]
    df = _normalize_strings(df, [col for col in string_cols if col in df.columns])

    if "runtime" in df.columns and "runtime_min" not in df.columns:
        df["runtime_min"] = df["runtime"].apply(_parse_runtime)
        df = df.drop(columns=["runtime"])

    for col in ["year", "rating", "votes", "gross", "runtime_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

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
        df["director"] = df["director"].fillna("").astype(str).str.strip()
        df["director_primary"] = ""
    if "star" in df.columns:
        df["star_primary"] = df["star"].fillna("").str.split(",").str[0].str.strip()

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
        df["_primary_genre"] = df["genre"].apply(_primary_genre)
        primary_runtime_median = df.groupby("_primary_genre")["runtime_min"].transform("median")
        global_runtime_median = df["runtime_min"].median()
        df["runtime_min"] = df["runtime_min"].fillna(primary_runtime_median).fillna(global_runtime_median)
        df = df.drop(columns=["_primary_genre"])

    if "gross" in df.columns and "year" in df.columns:
        df["gross"] = df.groupby("year")["gross"].transform(lambda series: series.fillna(series.median()))
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

    for col in ["movie_name", "genre", "description", "keywords", "star"]:
        if col in df.columns:
            literal_null_ratio = _literal_null_ratio(df[col])
            if literal_null_ratio > 0.001:
                raise ValueError(
                    f"Column {col} contains literal null tokens after normalization: ratio={literal_null_ratio:.4f}"
                )

    if "star" in df.columns:
        suspicious_cast_ratio = float(df["star"].apply(_row_has_suspicious_cast_fragment).mean())
        if suspicious_cast_ratio > 0.02:
            raise ValueError(
                f"Star column still contains suspicious cast fragments after parsing: ratio={suspicious_cast_ratio:.4f}"
            )

    if "movie_id" in df.columns and df["movie_id"].duplicated().any():
        raise ValueError("movie_id must remain unique after preprocessing.")

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Save the processed dataset to disk."""
    df.to_csv(output_path, index=False)
