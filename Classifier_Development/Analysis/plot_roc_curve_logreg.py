#!/usr/bin/env python3
"""
Generate ROC curve for Logistic Regression baseline classifier.

"""

import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datasets import load_dataset
from pathlib import Path

# Configuration
LOGREG_MODEL = Path("*/best_model.pkl")
GCS_TEST_DATA = "gs://*/full_test_data_yesno.jsonl"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "logreg_roc_curve.png"


def main():
    with open(LOGREG_MODEL, 'rb') as f:
        pipeline = pickle.load(f)

    local_test_file = "/tmp/test_data_yesno.jsonl"
    subprocess.run(
        ["gcloud", "storage", "cp", GCS_TEST_DATA, local_test_file],
        check=True, capture_output=True
    )
    test_dataset = load_dataset('json', data_files=local_test_file, split='train')

    # Prepare test data (response only, matching training)
    X_test = [sample['response'] for sample in test_dataset]
    y_test = [sample['label'] for sample in test_dataset]

    # Get predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Get class indices
    classes = list(pipeline.classes_)
    refusal_idx = classes.index('refusal')

    # Convert to binary
    y_true = np.array([1 if label == "refusal" else 0 for label in y_test])
    y_scores = y_proba[:, refusal_idx]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"ROC-AUC: {roc_auc:.4f}")

    # Find default threshold (θ=0.5)
    default_threshold = 0.5
    default_idx = np.argmin(np.abs(thresholds - default_threshold))
    default_tpr = tpr[default_idx]
    default_fpr = fpr[default_idx]

    # Find threshold where TPR >= 0.98
    recall_target = 0.98
    recall_candidates = np.where(tpr >= recall_target)[0]
    if len(recall_candidates) > 0:
        # Pick the one with lowest FPR among those meeting recall target
        best_recall_idx = recall_candidates[np.argmin(fpr[recall_candidates])]
        recall_threshold = thresholds[best_recall_idx]
        recall_tpr = tpr[best_recall_idx]
        recall_fpr = fpr[best_recall_idx]
    else:
        # Fallback: use maximum recall point
        best_recall_idx = np.argmax(tpr)
        recall_threshold = thresholds[best_recall_idx]
        recall_tpr = tpr[best_recall_idx]
        recall_fpr = fpr[best_recall_idx]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    # Main ROC curve
    plt.plot(
        fpr, tpr,
        color='#7c3aed',  # Purple for LogReg
        lw=2,
        label=f'Logistic Regression (AUC = {roc_auc:.4f})'
    )

    # Random classifier baseline
    plt.plot(
        [0, 1], [0, 1],
        color='gray',
        lw=1.5,
        linestyle='--',
        label='Random Classifier'
    )

    # Mark recall-optimized threshold
    plt.scatter(
        [recall_fpr], [recall_tpr],
        color='#dc2626',
        s=120,
        zorder=5,
        marker='o',
        label=f'θ={recall_threshold:.2f} (Recall-optimized)'
    )

    # Mark default threshold
    plt.scatter(
        [default_fpr], [default_tpr],
        color='#059669',
        s=100,
        zorder=5,
        marker='s',
        label=f'θ=0.5 (Default)'
    )

    # Place threshold info boxes side by side
    plt.text(
        0.70, 0.28,
        f'θ={recall_threshold:.2f} (Recall)\nTPR={recall_tpr:.2f}\nFPR={recall_fpr:.2f}',
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='#fee2e2', edgecolor='#dc2626', alpha=0.9)
    )

    plt.text(
        0.88, 0.28,
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
    plt.title('ROC Curve - Logistic Regression Baseline', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight')

    # Save PDF
    pdf_file = OUTPUT_DIR / "logreg_roc_curve.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')

    plt.close()

    # Calculate accuracy at default threshold
    accuracy = np.mean(y_pred == np.array(y_test))

if __name__ == "__main__":
    main()
