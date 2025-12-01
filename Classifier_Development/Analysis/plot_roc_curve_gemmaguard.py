#!/usr/bin/env python3
"""
Generate ROC curve for GemmaGuard.

"""

import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from pathlib import Path

# Configuration
GCS_RESULTS_PATH = "gs://*/final_results.json"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "gemmaguard_roc_curve.png"


def load_results_from_gcs():
    """Load final_results.json from GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "cat", GCS_RESULTS_PATH],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)


def main():
    # Load results
    results = load_results_from_gcs()
    evaluation = results["model"]["evaluation"]

    # Extract data
    ground_truth = evaluation["ground_truth"]
    probabilities = evaluation["probabilities"]

    # Convert to binary format
    y_true = np.array([1 if label == "refusal" else 0 for label in ground_truth])
    y_scores = np.array([p["refusal"] for p in probabilities])

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Verify AUC matches stored value
    stored_auc = evaluation["roc_auc"]

    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    # Find operating point at θ=0.2
    recall_threshold = 0.2
    recall_idx = np.argmin(np.abs(thresholds - recall_threshold))
    recall_tpr = tpr[recall_idx]
    recall_fpr = fpr[recall_idx]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    # Main ROC curve
    plt.plot(
        fpr, tpr,
        color='#2563eb',
        lw=2,
        label=f'GemmaGuard (AUC = {roc_auc:.3f})'
    )

    # Random classifier baseline
    plt.plot(
        [0, 1], [0, 1],
        color='gray',
        lw=1.5,
        linestyle='--',
        label='Random Classifier'
    )

    # Mark recall-optimized threshold (θ=0.2)
    plt.scatter(
        [recall_fpr], [recall_tpr],
        color='#dc2626',
        s=120,
        zorder=5,
        marker='o',
        label=f'θ=0.2 (Recall-optimized)'
    )

    # Mark default threshold (θ=0.5) for comparison
    plt.scatter(
        [optimal_fpr], [optimal_tpr],
        color='#059669',
        s=100,
        zorder=5,
        marker='s',
        label=f'θ=0.5 (Default)'
    )

    # Place threshold info boxes side by side above legend
    plt.text(
        0.70, 0.28,
        f'θ=0.2 (Recall)\nTPR={recall_tpr:.2f}\nFPR={recall_fpr:.2f}',
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9)
    )

    plt.text(
        0.88, 0.28,
        f'θ=0.5 (Default)\nTPR={optimal_tpr:.2f}\nFPR={optimal_fpr:.2f}',
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='#d1fae5', edgecolor='#059669', alpha=0.9)
    )

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - GemmaGuard Refusal Classifier', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {OUTPUT_FILE}")

    # Also save as PDF for LaTeX
    pdf_file = OUTPUT_DIR / "gemmaguard_roc_curve.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")

    plt.close()

if __name__ == "__main__":
    main()
