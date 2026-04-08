from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str = "Value",
    ylim_top: float | None = None,
    rotate_xt: int = 35,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#2c5282", edgecolor="#1a365d", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if rotate_xt:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xt, ha="right")
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blend_weights(path: Path, title: str, metrics: dict) -> bool:
    pairs = [
        ("content_weight", "content"),
        ("embedding_weight", "embedding"),
        ("genre_weight", "genre"),
        ("popularity_weight", "popularity"),
        ("bm25_weight", "bm25"),
    ]
    labels = [lab for key, lab in pairs if key in metrics]
    values = [float(metrics[key]) for key, _ in pairs if key in metrics]
    if not values:
        return False
    _bar_chart(
        path,
        title,
        labels,
        values,
        ylabel="Weight (normalized sum = 1)",
        ylim_top=1.05,
        rotate_xt=30,
    )
    return True


def plot_training_figures(fig_dir: Path, metrics: dict) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    timing_labels = ["fit (s)", "reranker (s)"]
    timing_vals = [
        float(metrics.get("fit_duration_sec", 0)),
        float(metrics.get("reranker_fit_duration_sec", 0)),
    ]
    p = fig_dir / "01_timing.png"
    _bar_chart(p, "Training — stage duration", timing_labels, timing_vals, ylabel="Seconds")
    saved.append(p)

    rows = float(metrics.get("training_rows", 0))
    p = fig_dir / "02_dataset_size.png"
    _bar_chart(p, "Training — dataset size", ["movies"], [rows], ylabel="Row count")
    saved.append(p)

    bw = fig_dir / "03_blend_weights.png"
    if _plot_blend_weights(bw, "Training — hybrid blend weights", metrics):
        saved.append(bw)

    rr_pairs = [
        ("reranker_sample_size", "rerank sample_size"),
        ("reranker_top_k", "rerank top_k"),
        ("min_votes", "min_votes"),
        ("faiss_top_k", "faiss top_k"),
    ]
    rr_labs = [lab for key, lab in rr_pairs if key in metrics and metrics[key] is not None]
    rr_vals = [float(metrics[key]) for key, _ in rr_pairs if key in metrics and metrics[key] is not None]
    if rr_vals:
        p = fig_dir / "04_reranker_and_thresholds.png"
        _bar_chart(p, "Training — reranker & retrieval thresholds", rr_labs, rr_vals, ylim_top=None)
        saved.append(p)

    flag_keys = ["use_embeddings", "use_bm25", "use_faiss"]
    flag_labs = [k.replace("_", " ") for k in flag_keys]
    flag_vals = [1.0 if metrics.get(k) else 0.0 for k in flag_keys]
    if any(k in metrics for k in flag_keys):
        p = fig_dir / "05_model_flags.png"
        _bar_chart(
            p,
            "Training — feature flags (1 = on)",
            flag_labs,
            flag_vals,
            ylabel="On/Off",
            ylim_top=1.15,
            rotate_xt=25,
        )
        saved.append(p)

    return saved


def plot_eval_figures(fig_dir: Path, metrics: dict) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    rank_keys = ["top1_hit", "precision_3", "recall_3", "ndcg_3", "ndcg_10"]
    rank_labels = ["top1 hit", "precision@3", "recall@3", "ndcg@3", "ndcg@10"]
    rank_vals = [float(metrics[k]) for k in rank_keys if k in metrics]
    rank_labs = [rank_labels[i] for i, k in enumerate(rank_keys) if k in metrics]
    if rank_vals:
        top = max(1.0, max(rank_vals) * 1.15)
        p = fig_dir / "01_ranking_metrics.png"
        _bar_chart(p, "Evaluation — offline ranking metrics", rank_labs, rank_vals, ylim_top=top)
        saved.append(p)

    bw = fig_dir / "02_blend_weights.png"
    if _plot_blend_weights(bw, "Evaluation — best blend weights", metrics):
        saved.append(bw)

    if "min_votes" in metrics:
        p = fig_dir / "03_min_votes.png"
        _bar_chart(
            p,
            "Evaluation — min_votes (best grid)",
            ["min_votes"],
            [float(metrics["min_votes"])],
            ylabel="Votes",
            ylim_top=None,
            rotate_xt=0,
        )
        saved.append(p)

    pair_slices = metrics.get("pair_slices", {})
    if pair_slices:
        slice_labels = list(pair_slices.keys())
        slice_vals = [float(pair_slices[name].get("top1_hit", 0.0)) for name in slice_labels]
        p = fig_dir / "04_pair_slice_top1.png"
        _bar_chart(
            p,
            "Evaluation — top1 by pair slice",
            slice_labels,
            slice_vals,
            ylabel="Top1 hit",
            ylim_top=1.0,
            rotate_xt=20,
        )
        saved.append(p)

    return saved


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "reports" / "results"
    training_json = results_dir / "training_metrics.json"
    eval_json = results_dir / "offline_metrics.json"
    training_fig = project_root / "reports" / "figures" / "training"
    eval_fig = project_root / "reports" / "figures" / "eval"

    if training_json.exists():
        training_metrics = json.loads(training_json.read_text())
        paths = plot_training_figures(training_fig, training_metrics)
        for p in paths:
            print(f"Saved: {p}")
    else:
        print(f"Skip training plots (missing {training_json})")

    if eval_json.exists():
        eval_metrics = json.loads(eval_json.read_text())
        paths = plot_eval_figures(eval_fig, eval_metrics)
        for p in paths:
            print(f"Saved: {p}")
    else:
        print(f"Skip eval plots (missing {eval_json})")


if __name__ == "__main__":
    main()
