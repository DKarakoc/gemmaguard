#!/usr/bin/env python3
"""
Download WildGuardMix dataset
"""

from datasets import load_dataset
import os
import json
from pathlib import Path
import requests


def check_dataset_access():
    """Check if dataset is publicly accessible or requires authentication"""
    

    # Try to get dataset info without downloading
    try:
        from huggingface_hub import dataset_info
        info = dataset_info("allenai/wildguardmix")
        print(f"Dataset found: {info}")
        print(f"Private: {info.private}")
        print(f"Gated: {info.gated}")
        return info
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        
    return None


def download_wildguardmix():
    """Try to download with authentication"""
    
    # Check if user has HF_TOKEN environment variable
    token = os.getenv('HF_TOKEN')
    if not token:
        print("No HF_TOKEN found in environment variables")
        print("To access this gated dataset:")
        print("1. Go to https://huggingface.co/datasets/allenai/wildguardmix")
        print("2. Request access if needed")
        print("3. Set HF_TOKEN environment variable with your token")
        print("4. Or run: huggingface-cli login")
        return None
    
    try:
        # Try with token
        dataset = load_dataset("allenai/wildguardmix", use_auth_token=token)
        
        save_path = Path("wildguardmix")
        save_path.mkdir(exist_ok=True)
        
        # Save dataset
        for split_name in dataset.keys():
            print(f"Processing {split_name} split")
            split_data = dataset[split_name]
            
            output_file = save_path / f"{split_name}.json"
            data_list = [item for item in split_data]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(data_list)} examples to {output_file}")
            
        return save_path
        
    except Exception as e:
        print(f"Authentication failed: {e}")
        return None


if __name__ == "__main__":
    # First check dataset access
    result = check_dataset_access()
    
    if result:
        print("Dataset access confirmed")
    else:
        print("Dataset requires authentication or is not publicly available")
        
    path = download_wildguardmix()
    
