#!/usr/bin/env python3
"""Generate confusion matrix heatmap figures for classifier comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent

CLASSIFIERS = {
    "gemmaguard": {
        "matrix": np.array([[507, 56], [89, 1068]]),
        "title": "GemmaGuard",
    },
    "wildguard": {
        "matrix": np.array([[550, 12], [133, 1023]]),
        "title": "WildGuard",
    },
    "logreg": {
        "matrix": np.array([[538, 25], [155, 1002]]),
        "title": "Logistic Regression",
    },
}

LABELS = ["Refusal", "Compliance"]


def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path) -> None:
    """Generate a heatmap-style confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create heatmap with Blues colormap
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)

    # Axis labels
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), ha="center")

    # Add text annotations with counts
    # Use white text on dark cells, black on light cells
    threshold = matrix.max() / 2.0
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=16,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    """Generate confusion matrix figures for all classifiers."""
    for name, data in CLASSIFIERS.items():
        output_path = OUTPUT_DIR / f"{name}_confusion_matrix.png"
        plot_confusion_matrix(data["matrix"], data["title"], output_path)


if __name__ == "__main__":
    main()
