from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess_movies, save_processed


def load_tmdb_csv(csv_path: Path, min_votes: int = 10) -> pd.DataFrame:
    """Load TMDB movies.csv into the canonical movie_id-based schema."""
    df = pd.read_csv(csv_path)
    print(f"Raw rows loaded: {len(df)}")

    df = df[df["status"] == "Released"].copy()
    df = df[df["vote_count"] >= min_votes]
    df = df[df["runtime"] > 0]
    df = df.dropna(subset=["title", "genres"])
    print(f"After quality filter (Released, votes>={min_votes}, has runtime+title+genres): {len(df)}")

    df = df.drop_duplicates(subset=["id"], keep="first")

    mapped = pd.DataFrame()
    mapped["movie_id"] = df["id"].astype(str)
    mapped["movie_name"] = df["title"]
    mapped["genre"] = df["genres"].str.replace("-", ", ", regex=False)
    mapped["rating"] = df["vote_average"]
    mapped["votes"] = df["vote_count"]
    mapped["description"] = df["overview"].fillna("")
    mapped["gross"] = df["revenue"]
    mapped["runtime_min"] = df["runtime"]

    mapped["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    credits = df["credits"].fillna("")
    mapped["star"] = credits.apply(
        lambda c: ", ".join(str(c).split("-")[:6]) if c else ""
    )
    mapped["director"] = ""

    mapped["keywords"] = (
        df["keywords"].fillna("").str.replace("-", ", ", regex=False)
    )
    mapped["poster_path"] = df["poster_path"].fillna("")
    mapped["popularity"] = df["popularity"]
    mapped["original_language"] = df["original_language"]
    mapped["tmdb_recommendations"] = df["recommendations"].fillna("")
    mapped["production_companies"] = (
        df["production_companies"].fillna("").str.replace("-", ", ", regex=False)
    )

    return mapped.reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "movies.csv"
    processed_path = project_root / "data" / "processed" / "movies_processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_tmdb_csv(raw_csv, min_votes=10)
    df = preprocess_movies(df)
    save_processed(df, str(processed_path))
    print(f"Saved {len(df)} movies to: {processed_path}")


if __name__ == "__main__":
    main()
