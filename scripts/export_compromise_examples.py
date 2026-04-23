from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.pairwise import build_pair_queries, classify_pair
from src.models.hybrid import HybridRecommender


LABEL_GUIDE = {
    "good_for_both": "Strong shared pick. Feels satisfying for both people, not just one side.",
    "acceptable_compromise": "Reasonable shared pick. Not ideal, but likely acceptable to both.",
    "leans_to_a": "Noticeably favors person A's taste more than person B's.",
    "leans_to_b": "Noticeably favors person B's taste more than person A's.",
    "bad_for_both": "Poor shared pick. Unlikely to satisfy either person as a compromise.",
}

ANNOTATION_RULE = (
    "Choose one label for each candidate. Prefer `good_for_both` only when the movie feels balanced and appealing "
    "for both seeds. Use `acceptable_compromise` for weaker but still plausible shared picks. Use `leans_to_a` or "
    "`leans_to_b` when the candidate clearly favors one side. Use `bad_for_both` when it does not work as a shared pick."
)


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


def _pair_lean_label(sim_a: object, sim_b: object, tolerance: float = 0.05) -> str:
    try:
        a = float(sim_a)
        b = float(sim_b)
    except (TypeError, ValueError):
        return "balanced"
    if a - b > tolerance:
        return "leans_to_a"
    if b - a > tolerance:
        return "leans_to_b"
    return "balanced"


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
        pair_type = classify_pair(df, idx_a, idx_b)
        response = model.recommend_date_movie(
            str(base_a["movie_id"]),
            str(base_b["movie_id"]),
            alternatives_n=max(args.candidates_per_pair - 1, 0),
            include_debug=True,
        )
        candidates = [response["best_pick"], *response["alternatives"]][: args.candidates_per_pair]
        for rank, candidate in enumerate(candidates, start=1):
            debug = candidate.get("debug_scores", {})
            explanation = candidate.get("explanation", {})
            rows.append(
                {
                    "annotation_rule": ANNOTATION_RULE,
                    "label_options": "|".join(LABEL_GUIDE.keys()),
                    "movie_id_a": str(base_a["movie_id"]),
                    "movie_name_a": str(base_a["movie_name"]),
                    "movie_genre_a": base_a.get("genre"),
                    "movie_year_a": None if pd.isna(base_a.get("year")) else int(base_a.get("year")),
                    "movie_id_b": str(base_b["movie_id"]),
                    "movie_name_b": str(base_b["movie_name"]),
                    "movie_genre_b": base_b.get("genre"),
                    "movie_year_b": None if pd.isna(base_b.get("year")) else int(base_b.get("year")),
                    "pair_type": pair_type,
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
                    "sim_mean": debug.get("sim_mean"),
                    "sim_gap": debug.get("sim_gap"),
                    "joint_score": debug.get("joint_score"),
                    "rule_score": debug.get("rule_score"),
                    "misery_penalty": debug.get("misery_penalty"),
                    "lean_hint": _pair_lean_label(debug.get("sim_a"), debug.get("sim_b")),
                    "works_for_both_hint": " | ".join(explanation.get("works_for_both", [])),
                    "leans_to_a_hint": " | ".join(explanation.get("leans_to_a", [])),
                    "leans_to_b_hint": " | ".join(explanation.get("leans_to_b", [])),
                    "label": "",
                    "label_confidence": "",
                    "label_notes": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} labeling rows to: {output_path}")
    print("Label guide:")
    print(json.dumps(LABEL_GUIDE, indent=2))


if __name__ == "__main__":
    main()
