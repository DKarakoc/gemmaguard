#!/usr/bin/env python3
"""
Logistic Regression Classification
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
from tqdm import tqdm

# Configuration
WILDGUARD_RESULTS_DIR = Path("wildguard_classified")
LOGREG_MODEL_PATH = Path("*/model.pkl")
OUTPUT_DIR = Path("logreg_classified")


def load_logreg_model(model_path: Path):
    """Load the trained Logistic Regression model."""

    if not model_path.exists():
        print(f"Error: LogReg model not found at {model_path}")
        exit(1)

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def get_english_response(result: Dict) -> str:
    """
    Extract English response text
    """
    if result["prompt_language"] == "turkish":
        return result.get("response_translation", "")
    else:
        return result.get("response", "")


def classify_with_logreg(pipeline, result: Dict) -> Dict:
    """
    Classify a single response with LogReg model.
    """
    # Get English text
    english_text = get_english_response(result)

    if not english_text:
        return {
            "logreg_refusal_probability": None,
            "logreg_compliance_probability": None,
            "logreg_prediction": None,
            "logreg_is_refusal": None,
            "error": "No English text available"
        }

    # Classify
    try:
        probabilities = pipeline.predict_proba([english_text])[0]
        prediction = pipeline.predict([english_text])[0]

        # Get probabilities
        classes = pipeline.classes_
        refusal_idx = np.where(classes == 'refusal')[0][0]
        compliance_idx = np.where(classes == 'compliance')[0][0]

        refusal_prob = float(probabilities[refusal_idx])

        return {
            "logreg_refusal_probability": refusal_prob,
            "logreg_compliance_probability": float(probabilities[compliance_idx]),
            "logreg_prediction": prediction,
            "logreg_is_refusal": (prediction == 'refusal'),
            "error": None
        }
    except Exception as e:
        return {
            "logreg_refusal_probability": None,
            "logreg_compliance_probability": None,
            "logreg_prediction": None,
            "logreg_is_refusal": None,
            "error": str(e)
        }


def process_file(filepath: Path, pipeline) -> Dict:
    """Process one WildGuard file and add LogReg classifications."""

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    source_lang = data["metadata"]["source_language"]
    prompt_lang = data["metadata"]["prompt_language"]
    n_samples = len(data["results"])

    # Classify each result
    classified_count = 0
    error_count = 0
    refusal_count = 0

    for result in tqdm(data["results"], desc="  Classifying"):
        logreg_result = classify_with_logreg(pipeline, result)

        # Add LogReg fields to result
        result["logreg_classification"] = logreg_result

        if logreg_result["error"] is None:
            classified_count += 1
            if logreg_result["logreg_is_refusal"]:
                refusal_count += 1
        else:
            error_count += 1

    print(f"Classified: {classified_count}/{n_samples}")
    if error_count > 0:
        print(f"Errors: {error_count}")

    if classified_count > 0:
        print(f" LogReg Refusal: {refusal_count}/{classified_count} ({100*refusal_count/classified_count:.1f}%)")
    else:
        print(f"No items classified (all failed)")
        return None  # Skip saving if no items were classified

    # Update metadata
    data["metadata"]["logreg_classified"] = True
    data["metadata"]["logreg_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    data["metadata"]["logreg_model_path"] = str(LOGREG_MODEL_PATH)
    data["metadata"]["logreg_stats"] = {
        "total": n_samples,
        "classified": classified_count,
        "refusal": refusal_count,
        "errors": error_count,
    }

    return data


def save_results(data: Dict, original_filepath: Path, output_dir: Path):
    """Save classified results with LogReg predictions."""
    # Create output filename
    output_filename = original_filepath.stem + "_logreg.json"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main execution."""

    start_time = datetime.now()

    # Check for WildGuard results
    if not WILDGUARD_RESULTS_DIR.exists():
        print(f"Error: WildGuard results directory not found")
        exit(1)

    wildguard_files = sorted(WILDGUARD_RESULTS_DIR.glob("*_wildguard_*.json"))

    if not wildguard_files:
        print(f"Error: No WildGuard classified files found in {WILDGUARD_RESULTS_DIR}")
        exit(1)

    # Load LogReg model
    pipeline = load_logreg_model(LOGREG_MODEL_PATH)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_errors = 0
    total_refusals = 0
    skipped_files = []

    for filepath in wildguard_files:
        data = process_file(filepath, pipeline)

        if data is None:
            # File had no classifiable items (empty responses)
            skipped_files.append(filepath.name)
            print(f"Skipped {filepath.name} (no classifiable items)\n")
            continue

        save_results(data, filepath, OUTPUT_DIR)

        # Count samples
        for result in data["results"]:
            total_samples += 1
            if result["logreg_classification"]["error"] is not None:
                total_errors += 1
            elif result["logreg_classification"]["logreg_is_refusal"]:
                total_refusals += 1

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if skipped_files:
        print("\nThe following files had no items to classify (empty responses):")
        for filename in skipped_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
