from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.pairwise import build_pair_queries
from src.models.hybrid import HybridRecommender


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pair recommendation examples for manual compromise labeling.")
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=project_root / "data" / "processed" / "movies_processed.csv",
        help="Path to the processed movies CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "reports" / "results" / "compromise_label_candidates.csv",
        help="Destination CSV for manual labeling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60,
        help="Number of pair queries to export.",
    )
    parser.add_argument(
        "--candidates-per-pair",
        type=int,
        default=5,
        help="How many recommendation candidates to export per pair.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = _parse_args(project_root)
    processed_path = args.processed_path
    output_path = args.output_path
    if not processed_path.is_absolute():
        processed_path = (project_root / processed_path).resolve()
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()

    artifact_path = project_root / "models" / "hybrid_artifacts.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}. Run scripts/train.py first.")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    df = pd.read_csv(processed_path)
    artifacts = joblib.load(artifact_path)
    model = HybridRecommender.from_artifacts(artifacts)
    pair_queries = build_pair_queries(df, sample_size=min(args.sample_size, len(df)), random_state=42)
    if not pair_queries:
        raise RuntimeError("No pair queries could be built from TMDB recommendations.")

    rows: list[dict[str, object]] = []
    for idx_a, idx_b in pair_queries:
        base_a = df.iloc[idx_a]
        base_b = df.iloc[idx_b]
        response = model.recommend_date_movie(
            str(base_a["movie_id"]),
            str(base_b["movie_id"]),
            alternatives_n=max(args.candidates_per_pair - 1, 0),
            include_debug=True,
        )
        candidates = [response["best_pick"], *response["alternatives"]][: args.candidates_per_pair]
        for rank, candidate in enumerate(candidates, start=1):
            debug = candidate.get("debug_scores", {})
            rows.append(
                {
                    "movie_id_a": str(base_a["movie_id"]),
                    "movie_name_a": str(base_a["movie_name"]),
                    "movie_id_b": str(base_b["movie_id"]),
                    "movie_name_b": str(base_b["movie_name"]),
                    "candidate_rank": rank,
                    "candidate_movie_id": candidate["movie_id"],
                    "candidate_movie_name": candidate["movie_name"],
                    "candidate_genre": candidate.get("genre"),
                    "candidate_year": candidate.get("year"),
                    "candidate_rating": candidate.get("rating"),
                    "shared_fit_score": candidate.get("shared_fit_score"),
                    "sim_a": debug.get("sim_a"),
                    "sim_b": debug.get("sim_b"),
                    "sim_min": debug.get("sim_min"),
                    "sim_gap": debug.get("sim_gap"),
                    "label": "",
                    "label_notes": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} labeling rows to: {output_path}")


if __name__ == "__main__":
    main()
