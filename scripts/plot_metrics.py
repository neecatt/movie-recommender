from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "reports" / "results" / "offline_metrics.json"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Offline Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(figures_dir / "offline_metrics.png", bbox_inches="tight")
    print(f"Saved plot to: {figures_dir / 'offline_metrics.png'}")


if __name__ == "__main__":
    main()
