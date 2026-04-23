from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.models.hybrid import HybridRecommender


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test the movie recommender with two movie inputs.")
    parser.add_argument(
        "--movie-a",
        required=True,
        help="Movie reference for person A. Use a movie_id or an exact title.",
    )
    parser.add_argument(
        "--movie-b",
        required=True,
        help="Movie reference for person B. Use a movie_id or an exact title.",
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=project_root / "models" / "hybrid_artifacts.joblib",
        help="Path to the trained model artifacts.",
    )
    parser.add_argument(
        "--alternatives",
        type=int,
        default=3,
        help="Number of alternative recommendations to show.",
    )
    parser.add_argument(
        "--include-debug",
        action="store_true",
        help="Include debug score fields in the output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full response as JSON instead of a human-readable summary.",
    )
    return parser.parse_args()


def _load_model(artifacts_path: Path) -> HybridRecommender:
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Model artifacts not found: {artifacts_path}. Run scripts/train.py first.")
    artifacts: dict[str, Any] = joblib.load(artifacts_path)
    HybridRecommender.validate_artifacts(artifacts)
    return HybridRecommender.from_artifacts(artifacts, validate=False)


def _format_movie_line(movie: dict[str, Any]) -> str:
    year = movie.get("year")
    rating = movie.get("rating")
    genre = movie.get("genre") or "Unknown genre"
    year_text = f" ({year})" if year is not None else ""
    rating_text = f" | rating {rating}" if rating is not None else ""
    return f"{movie['movie_name']}{year_text} | {genre}{rating_text}"


def _print_explanations(title: str, items: list[str]) -> None:
    print(title)
    if not items:
        print("- none")
        return
    for item in items:
        print(f"- {item}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    args = _parse_args(project_root)
    artifacts_path = args.artifacts_path
    if not artifacts_path.is_absolute():
        artifacts_path = (project_root / artifacts_path).resolve()

    model = _load_model(artifacts_path)
    response = model.recommend_date_movie(
        args.movie_a,
        args.movie_b,
        alternatives_n=max(args.alternatives, 0),
        include_debug=args.include_debug,
    )

    if args.json:
        print(json.dumps(response, indent=2))
        return

    print("Input Movies")
    print(f"- Person A: {_format_movie_line(response['movie_a'])}")
    print(f"- Person B: {_format_movie_line(response['movie_b'])}")
    print()

    print("Best Pick")
    print(f"- {_format_movie_line(response['best_pick'])}")
    print(f"- shared fit score: {response['best_pick']['shared_fit_score']:.4f}")
    print()

    explanation = response.get("explanation", {})
    _print_explanations("Why It Works For Both", explanation.get("works_for_both", []))
    print()
    _print_explanations("Why It Leans To Person A", explanation.get("leans_to_a", []))
    print()
    _print_explanations("Why It Leans To Person B", explanation.get("leans_to_b", []))

    if response.get("alternatives"):
        print()
        print("Alternatives")
        for idx, movie in enumerate(response["alternatives"], start=1):
            print(f"{idx}. {_format_movie_line(movie)} | shared fit score {movie['shared_fit_score']:.4f}")


if __name__ == "__main__":
    main()
