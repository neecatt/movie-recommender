from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess_movies, save_processed


def _word_count(value: str) -> int:
    return len([part for part in value.split() if part.strip()])


def _should_merge_cast_fragment(current: str, nxt: str, following: str | None) -> bool:
    """Repair surnames split by the raw hyphen delimiter.

    Example: ``Sergio Peris-Mencheta`` appears as ``Sergio Peris-Mencheta``
    in the raw field, but a naive split turns it into ``Sergio Peris`` and
    ``Mencheta``. We only merge when the middle token looks like an orphaned
    surname fragment between two multi-word names.
    """

    if not current or not nxt or following is None:
        return False
    if " " in nxt:
        return False
    if _word_count(current) < 2 or _word_count(following) < 2:
        return False
    if any(char.isdigit() for char in nxt):
        return False
    return True


def parse_cast(credits: object, max_cast: int = 6) -> str:
    text = str(credits or "").strip()
    if not text:
        return ""

    parts = [part.strip() for part in text.split("-") if part.strip()]
    if len(parts) <= 1:
        return text

    names: list[str] = []
    i = 0
    while i < len(parts):
        current = parts[i]
        following = parts[i + 1] if i + 1 < len(parts) else None
        after_following = parts[i + 2] if i + 2 < len(parts) else None
        if following is not None and _should_merge_cast_fragment(current, following, after_following):
            current = f"{current}-{following}"
            i += 1
        names.append(current)
        i += 1
        if len(names) >= max_cast:
            break

    return ", ".join(names)


def _base_tmdb_filter(df: pd.DataFrame, min_votes: int) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {"raw_rows": len(df)}

    released = df[df["status"] == "Released"].copy()
    counts["released_rows"] = len(released)

    voted = released[released["vote_count"] >= min_votes].copy()
    counts["vote_threshold_rows"] = len(voted)

    runtime_ok = voted[voted["runtime"] > 0].copy()
    counts["runtime_rows"] = len(runtime_ok)

    titled = runtime_ok.dropna(subset=["title", "genres"]).copy()
    counts["title_genre_rows"] = len(titled)

    deduped = titled.drop_duplicates(subset=["id"], keep="first").copy()
    counts["deduped_rows"] = len(deduped)

    return deduped, counts


def map_tmdb_df(df: pd.DataFrame, min_votes: int = 10) -> tuple[pd.DataFrame, dict[str, int | float | dict[str, bool]]]:
    """Map a raw TMDB DataFrame into the canonical movie_id-based schema."""

    filtered, counts = _base_tmdb_filter(df, min_votes=min_votes)

    mapped = pd.DataFrame()
    mapped["movie_id"] = filtered["id"].astype(str)
    mapped["movie_name"] = filtered["title"]
    mapped["genre"] = filtered["genres"].str.replace("-", ", ", regex=False)
    mapped["rating"] = filtered["vote_average"]
    mapped["votes"] = filtered["vote_count"]
    mapped["description"] = filtered["overview"].fillna("")
    mapped["gross"] = filtered["revenue"]
    mapped["runtime_min"] = filtered["runtime"]
    mapped["year"] = pd.to_datetime(filtered["release_date"], errors="coerce").dt.year
    mapped["star"] = filtered["credits"].fillna("").apply(parse_cast)
    mapped["director"] = ""
    mapped["keywords"] = filtered["keywords"].fillna("").str.replace("-", ", ", regex=False)
    mapped["poster_path"] = filtered["poster_path"].fillna("")
    mapped["popularity"] = filtered["popularity"]
    mapped["original_language"] = filtered["original_language"]
    mapped["tmdb_recommendations"] = filtered["recommendations"].fillna("")
    mapped["production_companies"] = (
        filtered["production_companies"].fillna("").str.replace("-", ", ", regex=False)
    )

    counts["director_available"] = False
    return mapped.reset_index(drop=True), counts


def load_tmdb_csv(csv_path: Path, min_votes: int = 10) -> tuple[pd.DataFrame, dict[str, int | float | dict[str, bool]]]:
    """Load TMDB movies.csv into the canonical movie_id-based schema."""
    df = pd.read_csv(csv_path)
    print(f"Raw rows loaded: {len(df)}")
    mapped, counts = map_tmdb_df(df, min_votes=min_votes)
    print(
        "After quality filter "
        f"(Released, votes>={min_votes}, has runtime+title+genres): {counts['deduped_rows']}"
    )
    return mapped, counts


def _build_summary(df: pd.DataFrame, counts: dict[str, int | float | dict[str, bool]]) -> dict[str, object]:
    coverage = {}
    for col in ["movie_name", "genre", "description", "keywords", "star", "director"]:
        if col in df.columns:
            coverage[col] = round(float(df[col].fillna("").astype(str).str.strip().ne("").mean()), 4)

    return {
        "row_counts": counts,
        "final_rows": len(df),
        "missing_counts": {
            "year": int(df["year"].isna().sum()) if "year" in df.columns else None,
            "runtime_min": int(df["runtime_min"].isna().sum()) if "runtime_min" in df.columns else None,
            "keywords": int(df["keywords"].fillna("").eq("").sum()) if "keywords" in df.columns else None,
            "star": int(df["star"].fillna("").eq("").sum()) if "star" in df.columns else None,
        },
        "coverage": coverage,
        "optional_features": {
            "director_available": bool(counts.get("director_available", False)),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "movies.csv"
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    summary_path = project_root / "reports" / "results" / "preprocess_summary.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df, counts = load_tmdb_csv(raw_csv, min_votes=10)
    df = preprocess_movies(df)
    save_processed(df, str(processed_path))

    summary = _build_summary(df, counts)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {len(df)} movies to: {processed_path}")


if __name__ == "__main__":
    main()
