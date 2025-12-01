#!/usr/bin/env python3
"""
Extract misclassification examples from Logistic Regression evaluation.

"""

import pickle
import subprocess
import numpy as np
from datasets import load_dataset
from pathlib import Path

# Configuration
LOGREG_MODEL = Path("*/model.pkl")
GCS_TEST_DATA = "gs://*/full_test_data_yesno.jsonl"
OUTPUT_DIR = Path("analysis")
OUTPUT_FILE = OUTPUT_DIR / "misclassification_examples_logreg.md"


def main():
    with open(LOGREG_MODEL, 'rb') as f:
        pipeline = pickle.load(f)

    local_test_file = "/tmp/test_data_yesno.jsonl"
    subprocess.run(
        ["gcloud", "storage", "cp", GCS_TEST_DATA, local_test_file],
        check=True, capture_output=True
    )
    test_dataset = load_dataset('json', data_files=local_test_file, split='train')

    # Prepare test data
    X_test = [f"{sample['prompt']} {sample['response']}" for sample in test_dataset]
    y_test = [sample['label'] for sample in test_dataset]

    # Get predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Get class indices
    classes = list(pipeline.classes_)
    refusal_idx = classes.index('refusal')

    # Find misclassifications
    false_positives = []
    false_negatives = []

    for i, (gt, pred, proba) in enumerate(zip(y_test, y_pred, y_proba)):
        if gt != pred:
            sample = test_dataset[i]
            refusal_prob = proba[refusal_idx]
            confidence = max(proba)

            error_info = {
                "index": i,
                "prompt": sample["prompt"],
                "response": sample["response"],
                "ground_truth": gt,
                "prediction": pred,
                "refusal_prob": refusal_prob,
                "confidence": confidence
            }

            if pred == "refusal" and gt == "compliance":
                false_positives.append(error_info)
            else:
                false_negatives.append(error_info)

    total_samples = len(y_test)
    total_errors = len(false_positives) + len(false_negatives)
    accuracy = (total_samples - total_errors) / total_samples * 100

    # Generate markdown report
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Logistic Regression Misclassification Analysis\n\n")
        f.write(f"**Total Test Samples**: {total_samples}\n")
        f.write(f"**Accuracy**: {accuracy:.2f}%\n")
        f.write(f"**Total Errors**: {total_errors}\n\n")

        f.write("## Key Observation\n\n")
        f.write(f"LogReg shows moderate **recall bias**: {len(false_positives)} FP vs {len(false_negatives)} FN.\n")
        f.write("It over-predicts refusals but less extremely than WildGuard.\n\n")

        # False Negatives
        f.write("## False Negatives (Missed Refusals)\n\n")
        f.write(f"*Model predicted compliance but response was actually a refusal ({len(false_negatives)} cases)*\n\n")

        # Sort by confidence (high confidence errors are more concerning)
        for i, err in enumerate(sorted(false_negatives, key=lambda x: -x["confidence"])[:5]):
            f.write(f"### FN Example {i+1} (Index: {err['index']})\n\n")
            f.write(f"**Confidence**: {err['confidence']:.3f} | **P(refusal)**: {err['refusal_prob']:.3f}\n\n")
            f.write("**User Request:**\n")
            f.write(f"```\n{err['prompt'][:500]}{'...' if len(err['prompt']) > 500 else ''}\n```\n\n")
            f.write("**Response (ACTUAL REFUSAL):**\n")
            f.write(f"```\n{err['response'][:500]}{'...' if len(err['response']) > 500 else ''}\n```\n\n")
            f.write("---\n\n")

        # False Positives
        f.write("## False Positives (Over-predicted Refusals)\n\n")
        f.write(f"*Model predicted refusal but response was actually compliance ({len(false_positives)} cases)*\n\n")

        for i, err in enumerate(sorted(false_positives, key=lambda x: -x["confidence"])[:5]):
            f.write(f"### FP Example {i+1} (Index: {err['index']})\n\n")
            f.write(f"**Confidence**: {err['confidence']:.3f} | **P(refusal)**: {err['refusal_prob']:.3f}\n\n")
            f.write("**User Request:**\n")
            f.write(f"```\n{err['prompt'][:500]}{'...' if len(err['prompt']) > 500 else ''}\n```\n\n")
            f.write("**Response (ACTUAL COMPLIANCE):**\n")
            f.write(f"```\n{err['response'][:500]}{'...' if len(err['response']) > 500 else ''}\n```\n\n")
            f.write("---\n\n")

        # Confidence analysis
        f.write("## Error Confidence Analysis\n\n")

        if false_negatives:
            avg_fn_conf = sum(e["confidence"] for e in false_negatives) / len(false_negatives)
            avg_fn_refusal = sum(e["refusal_prob"] for e in false_negatives) / len(false_negatives)
            f.write(f"**False Negatives**:\n")
            f.write(f"- Avg confidence: {avg_fn_conf:.3f}\n")
            f.write(f"- Avg P(refusal): {avg_fn_refusal:.3f}\n\n")

        if false_positives:
            avg_fp_conf = sum(e["confidence"] for e in false_positives) / len(false_positives)
            avg_fp_refusal = sum(e["refusal_prob"] for e in false_positives) / len(false_positives)
            f.write(f"**False Positives**:\n")
            f.write(f"- Avg confidence: {avg_fp_conf:.3f}\n")
            f.write(f"- Avg P(refusal): {avg_fp_refusal:.3f}\n\n")

        # Confidence distribution
        f.write("### Confidence Distribution of Errors\n\n")

        fn_high_conf = sum(1 for e in false_negatives if e["confidence"] > 0.9)
        fn_med_conf = sum(1 for e in false_negatives if 0.7 <= e["confidence"] <= 0.9)
        fn_low_conf = sum(1 for e in false_negatives if e["confidence"] < 0.7)

        fp_high_conf = sum(1 for e in false_positives if e["confidence"] > 0.9)
        fp_med_conf = sum(1 for e in false_positives if 0.7 <= e["confidence"] <= 0.9)
        fp_low_conf = sum(1 for e in false_positives if e["confidence"] < 0.7)

        f.write("| Confidence | FN | FP |\n")
        f.write("|------------|----|----|" + "\n")
        f.write(f"| High (>0.9) | {fn_high_conf} | {fp_high_conf} |\n")
        f.write(f"| Medium (0.7-0.9) | {fn_med_conf} | {fp_med_conf} |\n")
        f.write(f"| Low (<0.7) | {fn_low_conf} | {fp_low_conf} |\n")

    print(f"\nMisclassification report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
