"""
Prepare WildGuardMix dataset for training
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuration
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
RANDOM_SEED = 42

def format_sample(row):
    """
    Transform to training format with YES/NO answers.
    """
    # CRITICAL: Skip samples with NULL response_refusal_label
    if pd.isna(row['response_refusal_label']):
        return None

    prompt_text = f"""Classify if this response is a refusal or compliance.

User Request: {row['prompt']}

Assistant Response: {row['response']}

Is this a refusal?"""

    # Determine label (now guaranteed to be non-NULL)
    label = "refusal" if row['response_refusal_label'] == 'refusal' else "compliance"
    answer = "yes" if label == "refusal" else "no"

    return {
        "text": f"{prompt_text} {answer}",  # ← YES/NO FORMAT
        "label": label,
        "answer": answer,  # Added for clarity
        "prompt": row['prompt'],
        "response": row['response'],
        "category": row.get('prompt_harm_category', 'unknown'),
        "adversarial": row.get('adversarial', False)
    }


def main():

    # Load parquet files

    script_dir = Path(__file__).parent
    data_dir = script_dir / "wildguardmix"

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    train_parquet = data_dir / "wildguard_train.parquet"
    test_parquet = data_dir / "wildguard_test.parquet"

    if not test_parquet.exists():
        print(f"Error: Official test set not found at {test_parquet}")
        return

    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)


    # Filter for refusal task
    train_refusal_df = train_df[train_df['response'].notna() & (train_df['response'] != '')]
    test_refusal_df = test_df[test_df['response'].notna() & (test_df['response'] != '')]

    train_refusal_counts = train_refusal_df['response_refusal_label'].value_counts()
    for label, count in train_refusal_counts.items():
        pct = count / len(train_refusal_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    test_refusal_counts = test_refusal_df['response_refusal_label'].value_counts()
    for label, count in test_refusal_counts.items():
        pct = count / len(test_refusal_df) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # Transform train set to training format
    train_formatted_data = []
    train_null_count = 0
    for idx, row in train_refusal_df.iterrows():
        formatted = format_sample(row)
        if formatted is not None:
            train_formatted_data.append(formatted)
        else:
            train_null_count += 1

    if train_null_count > 0:
        print(f"Skipped {train_null_count} samples with NULL labels")
    print(f"Formatted {len(train_formatted_data):,} training samples")

    # Transform test set to training format
    test_formatted_data = []
    test_null_count = 0
    for idx, row in test_refusal_df.iterrows():
        formatted = format_sample(row)
        if formatted is not None:
            test_formatted_data.append(formatted)
        else:
            test_null_count += 1

    if test_null_count > 0:
        print(f"Skipped {test_null_count} samples with NULL labels (edge cases, unannotated)")
    print(f"Formatted {len(test_formatted_data):,} test samples ({len(test_refusal_df) - test_null_count} valid)")

    # Verify balance in train set
    train_labels = [d['label'] for d in train_formatted_data]
    train_label_counts = Counter(train_labels)
    print(f"\n  Train label distribution:")
    for label, count in train_label_counts.items():
        pct = count / len(train_labels) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # Use ALL data for training
    train_data = train_formatted_data
    val_data = []

    # Test data is already prepared from official test set
    test_data = test_formatted_data


    # Verify balance in each split
    for split_name, split_data in [("Training", train_data), ("Test", test_data)]:
        split_labels = [d['label'] for d in split_data]
        refusal_count = split_labels.count('refusal')
        compliance_count = split_labels.count('compliance')

    # Save to JSONL files
    output_dir = script_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)

    train_file = output_dir / "full_training_data_yesno.jsonl"
    test_file = output_dir / "full_test_data_yesno.jsonl"

    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(train_data):,} training samples to {train_file}")
    print(f"Saved {len(test_data):,} test samples to {test_file}")


    # Average text length
    train_lengths = [len(d['text']) for d in train_data]
    print(f"  Average text length: {sum(train_lengths)/len(train_lengths):.0f} chars")
    print(f"  Min text length: {min(train_lengths)} chars")
    print(f"  Max text length: {max(train_lengths)} chars")

    # Unique categories
    unique_cats = len(set(d['category'] for d in train_data))
    print(f"  Unique categories: {unique_cats}")

if __name__ == "__main__":
    main()
