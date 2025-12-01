#!/usr/bin/env python3
"""
Generate ROC curve for WildGuard with true probability AUC.

"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Configuration
PROBS_FILE = Path(__file__).parent.parent.parent / "*/wildguard_probs_summary_date_time.json"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "wildguard_roc_curve.png"


def main():
    # Load results
    with open(PROBS_FILE, 'r') as f:
        data = json.load(f)

    # Extract ROC curve data
    fpr = np.array(data['roc_curve']['fpr'])
    tpr = np.array(data['roc_curve']['tpr'])
    thresholds = np.array(data['roc_curve']['thresholds'])

    roc_auc = data['metrics']['roc_auc_probability']
    roc_auc_binary = data['metrics']['roc_auc_binary']

    recall_threshold = 0.2
    recall_idx = np.argmin(np.abs(thresholds - recall_threshold))
    recall_tpr = tpr[recall_idx]
    recall_fpr = fpr[recall_idx]

    # Find operating point at θ=0.5
    default_threshold = 0.5
    default_idx = np.argmin(np.abs(thresholds - default_threshold))
    default_tpr = tpr[default_idx]
    default_fpr = fpr[default_idx]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    # Main ROC curve
    plt.plot(
        fpr, tpr,
        color='#7c3aed',  # Purple for WildGuard
        lw=2,
        label=f'WildGuard (AUC = {roc_auc:.3f})'
    )

    # Random classifier baseline
    plt.plot(
        [0, 1], [0, 1],
        color='gray',
        lw=1.5,
        linestyle='--',
        label='Random Classifier'
    )

    # Mark recall-optimized threshold (θ=0.2) - RED like GemmaGuard
    plt.scatter(
        [recall_fpr], [recall_tpr],
        color='#dc2626',
        s=120,
        zorder=5,
        marker='o',
        label=f'θ=0.2 (Recall-optimized)'
    )

    # Mark default threshold (θ=0.5) - GREEN like GemmaGuard
    plt.scatter(
        [default_fpr], [default_tpr],
        color='#059669',
        s=100,
        zorder=5,
        marker='s',
        label=f'θ=0.5 (Default)'
    )

    # Place threshold info boxes (red/recall on left, green/default on right)
    plt.text(
        0.72, 0.28,
        f'θ=0.2 (Recall)\nTPR={recall_tpr:.2f}\nFPR={recall_fpr:.2f}',
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9)
    )

    plt.text(
        0.90, 0.28,
        f'θ=0.5 (Default)\nTPR={default_tpr:.2f}\nFPR={default_fpr:.2f}',
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='#d1fae5', edgecolor='#059669', alpha=0.9)
    )

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - WildGuard Refusal Classifier', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {OUTPUT_FILE}")

    # Also save as PDF for LaTeX
    pdf_file = OUTPUT_DIR / "wildguard_roc_curve.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")

    plt.close()

if __name__ == "__main__":
    main()
