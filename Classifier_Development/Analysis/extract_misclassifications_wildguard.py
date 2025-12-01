#!/usr/bin/env python3
"""
Extract misclassification examples from WildGuard evaluation.

"""

import json
from pathlib import Path

# Configuration
WILDGUARD_PREDICTIONS = Path("*/wildguard_predictions_inferenceclient_date_time.jsonl")
OUTPUT_DIR = Path("analysis")
OUTPUT_FILE = OUTPUT_DIR / "misclassification_examples_wildguard.md"


def main():
    predictions = []
    with open(WILDGUARD_PREDICTIONS) as f:
        for line in f:
            predictions.append(json.loads(line))

    # Sort by index
    predictions.sort(key=lambda x: x["index"])

    # Find misclassifications
    false_positives = []  # Predicted refusal, actually compliance
    false_negatives = []  # Predicted compliance, actually refusal

    for pred in predictions:
        gt = pred["true_label"]
        predicted = pred["predicted_refusal"]

        if gt != predicted:
            error_info = {
                "index": pred["index"],
                "prompt": pred["prompt"],
                "response": pred["response"],
                "ground_truth": gt,
                "prediction": predicted,
                "adversarial": pred.get("adversarial", False),
                "category": pred.get("category", "unknown")
            }

            if predicted == "refusal" and gt == "compliance":
                false_positives.append(error_info)
            else:
                false_negatives.append(error_info)

    total_samples = len(predictions)
    total_errors = len(false_positives) + len(false_negatives)
    accuracy = (total_samples - total_errors) / total_samples * 100

    # Generate markdown report
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# WildGuard Misclassification Analysis\n\n")
        f.write(f"**Total Test Samples**: {total_samples}\n")
        f.write(f"**Accuracy**: {accuracy:.2f}%\n")
        f.write(f"**Total Errors**: {total_errors}\n\n")

        f.write("## Key Observation\n\n")
        f.write(f"WildGuard shows a strong **recall bias**: {len(false_positives)} FP vs {len(false_negatives)} FN.\n")
        f.write("It heavily over-predicts refusals (97.86% recall) at the cost of precision (80.53%).\n\n")

        # False Negatives (rare but important)
        f.write("## False Negatives (Missed Refusals)\n\n")
        f.write(f"*Model predicted compliance but response was actually a refusal ({len(false_negatives)} cases)*\n\n")
        f.write("**Note**: WildGuard has very few FN due to its recall-optimized design.\n\n")

        for i, err in enumerate(false_negatives[:5]):
            f.write(f"### FN Example {i+1} (Index: {err['index']})\n\n")
            f.write(f"**Category**: {err['category']} | **Adversarial**: {err['adversarial']}\n\n")
            f.write("**User Request:**\n")
            f.write(f"```\n{err['prompt'][:500]}{'...' if len(err['prompt']) > 500 else ''}\n```\n\n")
            f.write("**Response (ACTUAL REFUSAL):**\n")
            f.write(f"```\n{err['response'][:500]}{'...' if len(err['response']) > 500 else ''}\n```\n\n")
            f.write("---\n\n")

        # False Positives (majority of errors)
        f.write("## False Positives (Over-predicted Refusals)\n\n")
        f.write(f"*Model predicted refusal but response was actually compliance ({len(false_positives)} cases)*\n\n")
        f.write("**Note**: This is WildGuard's primary error mode - flagging compliances as refusals.\n\n")

        for i, err in enumerate(false_positives[:5]):
            f.write(f"### FP Example {i+1} (Index: {err['index']})\n\n")
            f.write(f"**Category**: {err['category']} | **Adversarial**: {err['adversarial']}\n\n")
            f.write("**User Request:**\n")
            f.write(f"```\n{err['prompt'][:500]}{'...' if len(err['prompt']) > 500 else ''}\n```\n\n")
            f.write("**Response (ACTUAL COMPLIANCE):**\n")
            f.write(f"```\n{err['response'][:500]}{'...' if len(err['response']) > 500 else ''}\n```\n\n")
            f.write("---\n\n")

        # Category breakdown
        f.write("## Error Distribution by Category\n\n")

        fp_categories = {}
        for err in false_positives:
            cat = err["category"]
            fp_categories[cat] = fp_categories.get(cat, 0) + 1

        fn_categories = {}
        for err in false_negatives:
            cat = err["category"]
            fn_categories[cat] = fn_categories.get(cat, 0) + 1

        f.write("### False Positives by Category\n\n")
        for cat, count in sorted(fp_categories.items(), key=lambda x: -x[1])[:10]:
            f.write(f"- {cat}: {count}\n")

        if fn_categories:
            f.write("\n### False Negatives by Category\n\n")
            for cat, count in sorted(fn_categories.items(), key=lambda x: -x[1]):
                f.write(f"- {cat}: {count}\n")

        # Adversarial breakdown
        fp_adversarial = sum(1 for e in false_positives if e["adversarial"])
        fn_adversarial = sum(1 for e in false_negatives if e["adversarial"])

        f.write("\n## Adversarial vs Vanilla Errors\n\n")
        f.write(f"- FP on adversarial prompts: {fp_adversarial}/{len(false_positives)}\n")
        f.write(f"- FP on vanilla prompts: {len(false_positives) - fp_adversarial}/{len(false_positives)}\n")
        f.write(f"- FN on adversarial prompts: {fn_adversarial}/{len(false_negatives)}\n")
        f.write(f"- FN on vanilla prompts: {len(false_negatives) - fn_adversarial}/{len(false_negatives)}\n")

    print(f"\nMisclassification report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
