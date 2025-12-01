"""Data loading and sampling utilities."""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_offensive_samples(language: str, data_dir: Path = None) -> List[Dict]:
    """
    Load offensive tweet samples for a given language.

    Args:
        language: "turkish" or "english"
        data_dir: Path to data directory

    Returns:
        List of dicts with keys: id, text, language
    """
    filename = f"sample_offensive_{language}.jsonl"
    filepath = data_dir / filename

    if not filepath.exists():
        # Try legacy location
        legacy_path = data_dir / "sample_data" / filename
        if legacy_path.exists():
            filepath = legacy_path
        else:
            raise FileNotFoundError(
                f"Sample data not found: {filepath}\n"
            )

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def sample_tweets(
    language: str,
    n_samples: int,
    seed: int = 42,
    data_dir: Path = None
) -> List[Dict]:
    """
    Sample N offensive tweets for a language.

    Args:
        language: "turkish" or "english"
        n_samples: Number of samples to select
        seed: Random seed for reproducibility
        data_dir: Path to data directory

    Returns:
        List of sampled tweets
    """
    all_samples = load_offensive_samples(language, data_dir)

    if n_samples >= len(all_samples):
        return all_samples

    random.seed(seed)
    return random.sample(all_samples, n_samples)


def get_dataset_info(language: str = None, data_dir: Path = None) -> Dict:
    """
    Get information about available datasets.

    Args:
        language: Optional language filter
        data_dir: Path to data directory

    Returns:
        Dict with dataset statistics
    """

    info = {}

    languages = [language] if language else ["turkish", "english"]

    for lang in languages:
        try:
            samples = load_offensive_samples(lang, data_dir)
            info[lang] = {
                'total_samples': len(samples),
                'sample_ids': [s['id'] for s in samples[:5]],  # First 5 IDs
            }
        except FileNotFoundError:
            info[lang] = {'total_samples': 0, 'error': 'Data not found'}

    return info
