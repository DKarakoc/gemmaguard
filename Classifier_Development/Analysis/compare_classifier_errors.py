#!/usr/bin/env python3
"""
Compare error patterns across GemmaGuard, WildGuard, and Logistic Regression.

"""

import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Configuration
GCS_GEMMAGUARD_RESULTS = "gs://*/gemma3-270m-it/date_time/final_results.json"
WILDGUARD_PREDICTIONS = Path("wildguard_predictions_inferenceclient_date_time.jsonl")
LOGREG_MODEL = Path("*/logistic-regression-baseline/results/model.pkl")
OUTPUT_DIR = Path("analysis")


def load_gemmaguard_results():
    """Load GemmaGuard predictions from GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "cat", GCS_GEMMAGUARD_RESULTS],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    eval_data = data["model"]["evaluation"]
    return {
        "ground_truth": eval_data["ground_truth"],
        "predictions": eval_data["predictions"],
        "probabilities": eval_data["probabilities"]
    }


def load_wildguard_results():
    """Load WildGuard predictions from JSONL."""
    predictions = []
    with open(WILDGUARD_PREDICTIONS) as f:
        for line in f:
            predictions.append(json.loads(line))

    # Sort by index to ensure alignment
    predictions.sort(key=lambda x: x["index"])

    return {
        "ground_truth": [p["true_label"] for p in predictions],
        "predictions": [p["predicted_refusal"] for p in predictions],
        "data": predictions
    }


def load_logreg_results():
    """Load LogReg model and generate predictions with probabilities."""
    import pickle
    from datasets import load_dataset

    # Load model
    with open(LOGREG_MODEL, 'rb') as f:
        pipeline = pickle.load(f)

    # Load test data
    local_test_file = "/tmp/test_data_yesno.jsonl"
    subprocess.run(
        ["gcloud", "storage", "cp",
         "gs://*/full_test_data_yesno.jsonl",
         local_test_file],
        check=True, capture_output=True
    )
    test_dataset = load_dataset('json', data_files=local_test_file, split='train')

    # Prepare test texts (same format as training)
    X_test = [f"{sample['prompt']} {sample['response']}" for sample in test_dataset]
    y_test = [sample['label'] for sample in test_dataset]

    # Get predictions and probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Get class order
    classes = pipeline.classes_
    refusal_idx = list(classes).index('refusal')

    return {
        "ground_truth": y_test,
        "predictions": list(y_pred),
        "probabilities": [{"refusal": p[refusal_idx], "compliance": p[1-refusal_idx]}
                         for p in y_proba]
    }


def analyze_errors(name, ground_truth, predictions):
    """Analyze error patterns for a classifier."""
    errors = []
    for i, (gt, pred) in enumerate(zip(ground_truth, predictions)):
        if gt != pred:
            error_type = "FP" if pred == "refusal" else "FN"
            errors.append({"index": i, "type": error_type, "gt": gt, "pred": pred})

    fp = sum(1 for e in errors if e["type"] == "FP")
    fn = sum(1 for e in errors if e["type"] == "FN")

    return {
        "name": name,
        "total_errors": len(errors),
        "fp": fp,
        "fn": fn,
        "error_indices": set(e["index"] for e in errors),
        "fp_indices": set(e["index"] for e in errors if e["type"] == "FP"),
        "fn_indices": set(e["index"] for e in errors if e["type"] == "FN")
    }


def main():
    # Load all results
    gemmaguard = load_gemmaguard_results()
    wildguard = load_wildguard_results()
    logreg = load_logreg_results()

    # Verify alignment
    n_samples = len(gemmaguard["ground_truth"])
    assert len(wildguard["ground_truth"]) == n_samples, "Sample count mismatch with WildGuard"
    assert len(logreg["ground_truth"]) == n_samples, "Sample count mismatch with LogReg"

    # Verify ground truth consistency
    for i in range(n_samples):
        assert gemmaguard["ground_truth"][i] == wildguard["ground_truth"][i] == logreg["ground_truth"][i], \
            f"Ground truth mismatch at index {i}"

    # Analyze errors for each classifier
    gg_errors = analyze_errors("GemmaGuard", gemmaguard["ground_truth"], gemmaguard["predictions"])
    wg_errors = analyze_errors("WildGuard", wildguard["ground_truth"], wildguard["predictions"])
    lr_errors = analyze_errors("LogReg", logreg["ground_truth"], logreg["predictions"])

    for e in [gg_errors, wg_errors, lr_errors]:
        accuracy = (n_samples - e["total_errors"]) / n_samples * 100
        print(f"\n{e['name']}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Total Errors: {e['total_errors']} ({e['total_errors']/n_samples*100:.2f}%)")
        print(f"  False Positives: {e['fp']} (predicted refusal, was compliance)")
        print(f"  False Negatives: {e['fn']} (predicted compliance, was refusal)")

    # Samples where all three agree
    all_agree_correct = 0
    all_agree_wrong = 0

    # Pairwise agreement on errors
    gg_wg_both_wrong = gg_errors["error_indices"] & wg_errors["error_indices"]
    gg_lr_both_wrong = gg_errors["error_indices"] & lr_errors["error_indices"]
    wg_lr_both_wrong = wg_errors["error_indices"] & lr_errors["error_indices"]
    all_three_wrong = gg_errors["error_indices"] & wg_errors["error_indices"] & lr_errors["error_indices"]

    # Unique errors (only one classifier wrong)
    only_gg_wrong = gg_errors["error_indices"] - wg_errors["error_indices"] - lr_errors["error_indices"]
    only_wg_wrong = wg_errors["error_indices"] - gg_errors["error_indices"] - lr_errors["error_indices"]
    only_lr_wrong = lr_errors["error_indices"] - gg_errors["error_indices"] - wg_errors["error_indices"]


    # Where all three have FP
    all_fp = gg_errors["fp_indices"] & wg_errors["fp_indices"] & lr_errors["fp_indices"]
    all_fn = gg_errors["fn_indices"] & wg_errors["fn_indices"] & lr_errors["fn_indices"]


    # Analyze the samples where all three are wrong
    if all_three_wrong:

        # Load test data for examples
        wildguard_data = {p["index"]: p for p in wildguard["data"]}

        print("\nFirst 5 examples:")
        for i, idx in enumerate(sorted(list(all_three_wrong))[:5]):
            sample = wildguard_data[idx]
            gt = sample["true_label"]
            gg_pred = gemmaguard["predictions"][idx]

            print(f"\n--- Sample {idx} (labeled: {gt}, all predicted: {gg_pred}) ---")
            print(f"Prompt: {sample['prompt'][:150]}...")
            print(f"Response: {sample['response'][:200]}...")

if __name__ == "__main__":
    main()
