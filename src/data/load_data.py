from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


def iter_csv_rows(csv_path: Path) -> Iterable[dict]:
    """Yield rows from a CSV using the stdlib csv module for robustness."""
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row


def load_raw_csvs(data_dir: Path) -> pd.DataFrame:
    """Load and merge all CSVs under a directory into a single DataFrame."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dataframes: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = list(iter_csv_rows(csv_path))
        if rows:
            dataframes.append(pd.DataFrame.from_records(rows))

    if not dataframes:
        raise ValueError(f"No CSV files loaded from: {data_dir}")

    df = pd.concat(dataframes, ignore_index=True)
    df = df.drop_duplicates(subset=["movie_id"], keep="first").reset_index(drop=True)
    return df
