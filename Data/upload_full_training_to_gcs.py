"""
Upload FULL training dataset (YES/NO format) to GCS

Run with: uv run Data/upload_full_training_to_gcs.py

Uploads:
- full_training_data_yesno.jsonl
- full_test_data_yesno.jsonl
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run shell command and handle errors"""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(f"{description} complete")
    if result.stdout:
        print(result.stdout)
    return True


def main():
    # Get project ID
    result = subprocess.run(
        "gcloud config get-value project",
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error: Could not get project ID. Run 'gcloud auth login' first.")
        sys.exit(1)

    project_id = result.stdout.strip()
    if not project_id:
        print("Error: No project configured. Run 'gcloud config set project YOUR_PROJECT_ID'")
        sys.exit(1)

    print(f"  Project ID: {project_id}")
    bucket_name = f"{project_id}-gemma-quickstart"
    print(f"  Bucket: gs://{bucket_name}")

    # Check if data files exist
    script_dir = Path(__file__).parent
    data_dir = script_dir / "processed_data"

    files_to_upload = [
        ("full_training_data_yesno.jsonl", "Training data"),
        ("full_test_data_yesno.jsonl", "Test data")
    ]

    for filename, desc in files_to_upload:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"Error: {file_path} not found")
            sys.exit(1)

        size_mb = file_path.stat().st_size / (1024 * 1024)

    # Check if bucket exists
    result = subprocess.run(
        f"gcloud storage buckets describe gs://{bucket_name}",
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Bucket does not exist. Creating")
        if not run_command(
            f"gcloud storage buckets create gs://{bucket_name} --location=us-central1",
            "Creating bucket"
        ):
            sys.exit(1)
    else:
        print(f"Bucket exists")

    # Upload files
    upload_commands = []
    for filename, desc in files_to_upload:
        local_path = data_dir / filename
        gcs_path = f"gs://{bucket_name}/data/full/{filename}"
        upload_commands.append((
            f"gcloud storage cp {local_path} {gcs_path}",
            f"Uploading {desc}"
        ))

    for cmd, desc in upload_commands:
        if not run_command(cmd, desc):
            sys.exit(1)

    # Verify upload
    run_command(
        f"gcloud storage ls gs://{bucket_name}/data/full/*yesno.jsonl",
        "Checking uploaded yes/no files"
    )

    # Calculate total size
    total_size = sum((data_dir / f).stat().st_size for f, _ in files_to_upload)
    total_size_mb = total_size / (1024 * 1024)

    # Summary
    print("UPLOAD COMPLETE!")
    print(f"  Training:   gs://{bucket_name}/data/full/full_training_data_yesno.jsonl")
    print(f"  Test:       gs://{bucket_name}/data/full/full_test_data_yesno.jsonl")

if __name__ == "__main__":
    main()
