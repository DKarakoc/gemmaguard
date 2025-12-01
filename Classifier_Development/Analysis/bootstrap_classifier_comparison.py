#!/usr/bin/env python3
"""
Bootstrap Confidence Interval Analysis Classifiers.

Compares GemmaGuard, WildGuard, and Logistic Regression using bootstrapped
confidence intervals
"""

import json
import subprocess
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets import load_dataset

# Configuration
GCS_GEMMAGUARD_RESULTS = "gs://*/final_results.json"
WILDGUARD_PREDICTIONS = Path("data/wildguard_predictions_inferenceclient_day_time.jsonl")
LOGREG_MODEL = Path("*/logistic-regression-baseline/results/model.pkl")
OUTPUT_DIR = Path("analysis/results")

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def load_gemmaguard_results():
    """Load GemmaGuard predictions from GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "cat", GCS_GEMMAGUARD_RESULTS],
        capture_output=True, text=True, check=True
    )
    data = json.loads(result.stdout)
    eval_data = data["model"]["evaluation"]

    # Convert to binary (1 = refusal, 0 = compliance)
    ground_truth = [1 if g == "refusal" else 0 for g in eval_data["ground_truth"]]
    predictions = [1 if p == "refusal" else 0 for p in eval_data["predictions"]]

    return {
        "name": "GemmaGuard",
        "ground_truth": np.array(ground_truth),
        "predictions": np.array(predictions),
    }


def load_wildguard_results():
    """Load WildGuard predictions from JSONL."""
    predictions_data = []
    with open(WILDGUARD_PREDICTIONS) as f:
        for line in f:
            predictions_data.append(json.loads(line))

    # Sort by index to ensure alignment
    predictions_data.sort(key=lambda x: x["index"])

    # Convert to binary (predicted_refusal is a string: "refusal" or "compliance")
    ground_truth = [1 if p["true_label"] == "refusal" else 0 for p in predictions_data]
    predictions = [1 if p["predicted_refusal"] == "refusal" else 0 for p in predictions_data]

    return {
        "name": "WildGuard",
        "ground_truth": np.array(ground_truth),
        "predictions": np.array(predictions),
    }


def load_logreg_results():
    """Load LogReg model and generate predictions."""

    # Load model
    with open(LOGREG_MODEL, 'rb') as f:
        pipeline = pickle.load(f)

    # Load test data
    local_test_file = "/tmp/test_data_yesno.jsonl"
    subprocess.run(
        ["gcloud", "storage", "cp",
         "gs://*/data/full/full_test_data_yesno.jsonl",
         local_test_file],
        check=True, capture_output=True
    )
    test_dataset = load_dataset('json', data_files=local_test_file, split='train')

    # Prepare test texts (same format as training)
    X_test = [f"{sample['prompt']} {sample['response']}" for sample in test_dataset]
    y_test = [sample['label'] for sample in test_dataset]

    # Get predictions
    y_pred = pipeline.predict(X_test)

    # Convert to binary
    ground_truth = [1 if y == "refusal" else 0 for y in y_test]
    predictions = [1 if p == "refusal" else 0 for p in y_pred]

    return {
        "name": "Logistic Regression",
        "ground_truth": np.array(ground_truth),
        "predictions": np.array(predictions),
    }


def bootstrap_accuracy(ground_truth: np.ndarray, predictions: np.ndarray,
                       n_bootstrap: int = N_BOOTSTRAP,
                       random_seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Compute bootstrap distribution of accuracy.

    Args:
        ground_truth: Array of true labels (0/1)
        predictions: Array of predicted labels (0/1)
        n_bootstrap: Number of bootstrap resamples
        random_seed: Random seed for reproducibility

    Returns:
        Array of n_bootstrap accuracy values
    """
    np.random.seed(random_seed)
    n_samples = len(ground_truth)
    accuracies = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        gt_resample = ground_truth[indices]
        pred_resample = predictions[indices]

        # Calculate accuracy for this resample
        accuracies[i] = accuracy_score(gt_resample, pred_resample)

    return accuracies


def compute_confidence_interval(bootstrap_distribution: np.ndarray,
                                confidence_level: float = CONFIDENCE_LEVEL) -> dict:
    """
    Compute confidence interval from bootstrap distribution using percentile method.

    Args:
        bootstrap_distribution: Array of bootstrap statistics
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dictionary with CI bounds and statistics
    """
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100  # 2.5 for 95% CI
    upper_percentile = (1 - alpha / 2) * 100  # 97.5 for 95% CI

    ci_lower = np.percentile(bootstrap_distribution, lower_percentile)
    ci_upper = np.percentile(bootstrap_distribution, upper_percentile)

    return {
        "mean": np.mean(bootstrap_distribution),
        "std": np.std(bootstrap_distribution),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "median": np.median(bootstrap_distribution),
    }


def create_boxplot(results: dict, output_path: Path):
    """
    Create boxplot visualization of bootstrap distributions.

    Args:
        results: Dictionary with classifier names as keys, bootstrap distributions as values
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for boxplot
    names = list(results.keys())
    data = [results[name]["bootstrap_distribution"] * 100 for name in names]  # Convert to percentage

    # Create boxplot
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8))

    # Color the boxes
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Bootstrap Distribution of Classifier Accuracy\n(n={N_BOOTSTRAP} resamples, 95% CI)', fontsize=14)

    # Add CI annotations
    for i, name in enumerate(names):
        ci = results[name]["confidence_interval"]
        ci_text = f'95% CI: [{ci["ci_lower"]*100:.2f}%, {ci["ci_upper"]*100:.2f}%]'
        ax.annotate(ci_text, xy=(i+1, ci["ci_lower"]*100 - 1.5),
                    ha='center', fontsize=9, color='darkblue')

    # Adjust y-axis to show all annotations
    y_min = min([results[name]["confidence_interval"]["ci_lower"] for name in names]) * 100 - 4
    y_max = max([results[name]["confidence_interval"]["ci_upper"] for name in names]) * 100 + 2
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save figure (high resolution for print quality)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def save_results_json(results: dict, output_path: Path):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            "point_estimate_accuracy": float(data["point_estimate"]),
            "confidence_interval": {
                "ci_lower": float(data["confidence_interval"]["ci_lower"]),
                "ci_upper": float(data["confidence_interval"]["ci_upper"]),
                "ci_width": float(data["confidence_interval"]["ci_width"]),
                "mean": float(data["confidence_interval"]["mean"]),
                "std": float(data["confidence_interval"]["std"]),
                "median": float(data["confidence_interval"]["median"]),
            },
            "n_samples": int(data["n_samples"]),
        }

    output = {
        "metadata": {
            "n_bootstrap": N_BOOTSTRAP,
            "confidence_level": CONFIDENCE_LEVEL,
            "random_seed": RANDOM_SEED,
            "methodology": "Percentile bootstrap (Efron & Tibshirani, 1993)",
        },
        "results": json_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved JSON results to {output_path}")


def main():
    """Main execution."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all classifier results
    classifiers = [
        load_gemmaguard_results(),
        load_wildguard_results(),
        load_logreg_results(),
    ]

    # Verify alignment
    n_samples = len(classifiers[0]["ground_truth"])

    for clf in classifiers:
        assert len(clf["ground_truth"]) == n_samples, f"{clf['name']} has different sample count!"
        assert len(clf["predictions"]) == n_samples, f"{clf['name']} has different prediction count!"

    # Compute bootstrap distributions
    results = {}

    for clf in classifiers:
        print(f"\nComputing bootstrap distribution for {clf['name']}...")

        # Point estimate
        point_estimate = accuracy_score(clf["ground_truth"], clf["predictions"])

        # Bootstrap distribution
        bootstrap_dist = bootstrap_accuracy(clf["ground_truth"], clf["predictions"])

        # Confidence interval
        ci = compute_confidence_interval(bootstrap_dist)

        results[clf["name"]] = {
            "point_estimate": point_estimate,
            "bootstrap_distribution": bootstrap_dist,
            "confidence_interval": ci,
            "n_samples": n_samples,
        }


    # Create boxplot
    boxplot_path = OUTPUT_DIR / "bootstrap_classifier_comparison.png"
    create_boxplot(results, boxplot_path)

    # Save JSON results
    json_path = OUTPUT_DIR / "bootstrap_classifier_comparison.json"
    save_results_json(results, json_path)

if __name__ == "__main__":
    main()
