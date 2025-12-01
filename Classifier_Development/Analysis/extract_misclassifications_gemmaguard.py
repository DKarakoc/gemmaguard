#!/usr/bin/env python3
"""
Extract misclassification examples from GemmaGuard evaluation.

"""

import json
import subprocess
from datasets import load_dataset
from pathlib import Path

# Configuration
GCS_RESULTS_PATH = "gs://*/final_results.json"
GCS_TEST_DATA = "gs://*/full_test_data_yesno.jsonl"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "misclassification_examples.md"


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
    results = load_results_from_gcs()
    evaluation = results["model"]["evaluation"]

    ground_truth = evaluation["ground_truth"]
    predictions = evaluation["predictions"]
    probabilities = evaluation["probabilities"]

    # Download test data locally first
    local_test_file = "/tmp/test_data_yesno.jsonl"
    subprocess.run(
        ["gcloud", "storage", "cp", GCS_TEST_DATA, local_test_file],
        check=True,
        capture_output=True
    )
    test_dataset = load_dataset('json', data_files=local_test_file, split='train')

    # Find misclassifications
    false_positives = []  # Predicted refusal, actually compliance
    false_negatives = []  # Predicted compliance, actually refusal

    for i, (gt, pred, prob) in enumerate(zip(ground_truth, predictions, probabilities)):
        if gt != pred:
            sample = test_dataset[i]
            error_info = {
                "index": i,
                "prompt": sample["prompt"],
                "response": sample["response"],
                "ground_truth": gt,
                "prediction": pred,
                "refusal_prob": prob["refusal"],
                "confidence": prob["confidence"]
            }

            if pred == "refusal" and gt == "compliance":
                false_positives.append(error_info)
            else:
                false_negatives.append(error_info)

    # Generate markdown report
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# GemmaGuard Misclassification Analysis\n\n")
        f.write(f"**Total Test Samples**: {len(ground_truth)}\n")
        f.write(f"**Accuracy**: {evaluation['accuracy']*100:.2f}%\n")
        f.write(f"**Total Errors**: {len(false_positives) + len(false_negatives)}\n\n")

        # False Negatives (more critical - missed refusals)
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
        f.write("## False Positives (Incorrect Refusal Labels)\n\n")
        f.write(f"*Model predicted refusal but response was actually compliance ({len(false_positives)} cases)*\n\n")

        for i, err in enumerate(sorted(false_positives, key=lambda x: -x["confidence"])[:5]):
            f.write(f"### FP Example {i+1} (Index: {err['index']})\n\n")
            f.write(f"**Confidence**: {err['confidence']:.3f} | **P(refusal)**: {err['refusal_prob']:.3f}\n\n")
            f.write("**User Request:**\n")
            f.write(f"```\n{err['prompt'][:500]}{'...' if len(err['prompt']) > 500 else ''}\n```\n\n")
            f.write("**Response (ACTUAL COMPLIANCE):**\n")
            f.write(f"```\n{err['response'][:500]}{'...' if len(err['response']) > 500 else ''}\n```\n\n")
            f.write("---\n\n")

        # Summary statistics
        f.write("## Error Analysis Summary\n\n")

        if false_negatives:
            avg_fn_conf = sum(e["confidence"] for e in false_negatives) / len(false_negatives)
            f.write(f"**False Negatives**: Avg confidence = {avg_fn_conf:.3f}\n")

        if false_positives:
            avg_fp_conf = sum(e["confidence"] for e in false_positives) / len(false_positives)
            f.write(f"**False Positives**: Avg confidence = {avg_fp_conf:.3f}\n")

    print(f"\nMisclassification report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
