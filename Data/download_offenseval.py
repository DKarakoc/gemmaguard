from huggingface_hub import snapshot_download
import os
import pandas as pd

repo_id = "strombergnlp/offenseval_2020"

print("OffensEval 2020 Dataset Download")
print("NOTE: This dataset may require authentication.")
print("If the download fails, you may need to:")
print("1. Create a Hugging Face account at https://huggingface.co")
print("2. Accept the dataset terms at https://huggingface.co/datasets/strombergnlp/offenseval_2020")
print("3. Login using: huggingface-cli login")

print("\nAttempting to download the entire dataset repository")

try:
    # Try to download the entire repository
    local_dir = snapshot_download(
        repo_id=repo_id, 
        repo_type="dataset",
        local_dir="offenseval_2020_full",
        ignore_patterns=["*.py", "*.md", "*.json"]  # Skip script files
    )
    print(f"Dataset downloaded to: {local_dir}")
    
    # List downloaded files
    print("\nFiles downloaded:")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, local_dir)
            print(f"  - {rel_path}")
            
            # Try to read TSV files
            if file.endswith('.tsv'):
                try:
                    df = pd.read_csv(filepath, sep='\t')
                    print(f"    Shape: {df.shape}")
                    print(f"    Columns: {list(df.columns)}")
                    if len(df) > 0:
                        print(f"    First row: {df.iloc[0].to_dict()}")
                except Exception as e:
                    print(f"    Error reading file: {e}")
                    
except Exception as e:
    print(f"\nError: {e}")
    print("\nAlternative approach: Manual download")
    print("\nYou can manually download the dataset:")
    print("https://huggingface.co/datasets/strombergnlp/offenseval_2020")
