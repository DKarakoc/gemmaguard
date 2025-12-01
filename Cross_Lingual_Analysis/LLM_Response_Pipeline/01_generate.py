#!/usr/bin/env python3
"""
Multi-Model Cross-Lingual Safety Study
"""

import sys
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from shared.llm_clients import OpenRouterClient, OpenAIClient

# Load environment
load_dotenv()

# Configuration
MODELS = [
    # OpenAI models
    {
        "id": "gpt-5-nano",
        "name": "GPT-5-nano",
        "provider": "openai",
        "turkish_support": "OFFICIAL",
        "rate_limit_delay": 0,  # No delay
    },
    {
        "id": "gpt-4o-2024-05-13",
        "name": "GPT-4o",
        "provider": "openai",
        "turkish_support": "OFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "gpt-4o-mini-2024-07-18",
        "name": "GPT-4o-mini",
        "provider": "openai",
        "turkish_support": "OFFICIAL",
        "rate_limit_delay": 0,
    },
    # OpenRouter models
    {
        "id": "mistralai/mixtral-8x22b-instruct",
        "name": "Mixtral 8x22B",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "deepseek/deepseek-chat-v3-0324",
        "name": "DeepSeek V3",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Qwen2.5 72B",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash",
        "provider": "openrouter",
        "turkish_support": "OFFICIAL",
        "rate_limit_delay": 0,
    },
    {
        "id": "deepseek/deepseek-chat-v3.1",
        "name": "DeepSeek V3.1",
        "provider": "openrouter",
        "turkish_support": "UNOFFICIAL",
        "rate_limit_delay": 0,
    },
]


SAMPLES_PER_SOURCE = 1000
STRATEGY = "implicit_raw_controlled_factorial"
SEED_TURKISH_SOURCED = 5000
SEED_ENGLISH_SOURCED = 5001

TURKISH_SOURCED_FILE = Path("offenseval_tr_to_en_train_FINAL.jsonl")
ENGLISH_SOURCED_FILE = Path("offenseval_en_to_tr_train_FINAL.jsonl")
OUTPUT_DIR = Path("raw_responses")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_time_remaining(n_completed, n_total, start_time):
    """Estimate ETA based on current progress."""
    if n_completed == 0:
        return "calculating"
    elapsed = time.time() - start_time
    avg_time_per_sample = elapsed / n_completed
    remaining_samples = n_total - n_completed
    eta_seconds = remaining_samples * avg_time_per_sample
    return str(timedelta(seconds=int(eta_seconds)))


def load_translation_pairs(filepath: Path, n_samples: int, seed: int, source_language: str) -> List[Dict]:
    """Load translation pairs from FINAL manual translation JSONL files.

    Args:
        filepath: Path to FINAL translation file
        n_samples: Number of pairs to sample
        seed: Random seed for reproducibility
        source_language: "turkish" or "english" - which language is the authentic original

    Returns:
        List of dicts with id, turkish_text, english_text, source_language
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Manual translation file not found: {filepath}\n"
        )

    # Load all pairs
    all_pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # Determine which is original and which is translation
            if source_language == "turkish":
                # Turkish original → English translation
                turkish_text = item["original"]
                english_text = item["translation"]
            else:  # source_language == "english"
                # English original → Turkish translation
                english_text = item["original"]
                turkish_text = item["translation"]

            all_pairs.append({
                "id": item["id"],
                "turkish_text": turkish_text,
                "english_text": english_text,
                "source_language": source_language,
                "notes": item.get("notes", ""),
            })

    # Sample n_samples randomly
    if n_samples >= len(all_pairs):
        sampled = all_pairs
    else:
        random.seed(seed)
        sampled = random.sample(all_pairs, n_samples)
    return sampled


def create_implicit_prompt(tweet: str) -> str:
    """Create implicit prompt - just the raw tweet, NO instruction."""
    return tweet


def generate_responses_for_condition(
    model: Dict,
    pairs: List[Dict],
    source_language: str
) -> Dict:
    """Generate responses for one source language condition.

    Args:
        model: Model configuration dict
        pairs: List of translation pairs
        source_language: "turkish" or "english"

    Returns:
        Dict with turkish and english results
    """

    # Initialize client based on provider
    if model["provider"] == "openai":
        client = OpenAIClient(
            model=model["id"],
            max_tokens=4096,
            temperature=0.7
        )
    else:  # openrouter
        client = OpenRouterClient(
            model=model["id"],
            max_tokens=4096,
            temperature=0.7
        )

    results_english = []
    results_turkish = []
    total_cost = 0.0
    start_time = time.time()

    # Generate for each pair
    for i, pair in enumerate(tqdm(pairs, desc=f"  {source_language}-sourced")):
        pair_id = pair["id"]

        # Apply rate limiting delay if needed
        if model["rate_limit_delay"] > 0 and i > 0:
            time.sleep(model["rate_limit_delay"])

        # TURKISH PROMPT
        turkish_prompt = create_implicit_prompt(pair["turkish_text"])
        turkish_response = client.generate(turkish_prompt)

        if "estimated_cost" in turkish_response:
            total_cost += turkish_response["estimated_cost"]

        results_turkish.append({
            "sample_id": pair_id,
            "pair_id": pair_id,
            "source_language": source_language,
            "prompt_language": "turkish",
            "is_authentic": (source_language == "turkish"),
            "original_tweet": pair["turkish_text"],
            "matched_translation": pair["english_text"],
            "translation_notes": pair.get("notes", ""),
            "model": model["name"],
            "model_id": model["id"],
            "provider": model["provider"],
            "turkish_support": model["turkish_support"],
            "strategy": STRATEGY,
            "prompt": turkish_prompt,
            "system_prompt": None,
            "response": turkish_response["response"],
            "finish_reason": turkish_response["finish_reason"],
            "error": turkish_response["error"],
            "reasoning_summary": turkish_response.get("reasoning_summary", []),
            "tokens": {
                "prompt": turkish_response["prompt_tokens"],
                "completion": turkish_response["completion_tokens"],
                "total": turkish_response["total_tokens"],
            },
            "cost": turkish_response.get("estimated_cost", 0.0),
        })

        # Apply rate limiting delay before next request
        if model["rate_limit_delay"] > 0:
            time.sleep(model["rate_limit_delay"])

        # ENGLISH PROMPT
        english_prompt = create_implicit_prompt(pair["english_text"])
        english_response = client.generate(english_prompt)

        if "estimated_cost" in english_response:
            total_cost += english_response["estimated_cost"]

        results_english.append({
            "sample_id": pair_id,
            "pair_id": pair_id,
            "source_language": source_language,
            "prompt_language": "english",
            "is_authentic": (source_language == "english"),
            "original_tweet": pair["english_text"],
            "matched_translation": pair["turkish_text"],
            "translation_notes": pair.get("notes", ""),
            "model": model["name"],
            "model_id": model["id"],
            "provider": model["provider"],
            "turkish_support": model["turkish_support"],
            "strategy": STRATEGY,
            "prompt": english_prompt,
            "system_prompt": None,
            "response": english_response["response"],
            "finish_reason": english_response["finish_reason"],
            "error": english_response["error"],
            "reasoning_summary": english_response.get("reasoning_summary", []),
            "tokens": {
                "prompt": english_response["prompt_tokens"],
                "completion": english_response["completion_tokens"],
                "total": english_response["total_tokens"],
            },
            "cost": english_response.get("estimated_cost", 0.0),
        })

        # Show progress with ETA every 50 samples
        if (i + 1) % 50 == 0:
            eta = estimate_time_remaining((i + 1) * 2, len(pairs) * 2, start_time)
            print(f"    Progress: {(i + 1) * 2}/{len(pairs) * 2} - ETA: {eta} - Cost so far: ${total_cost:.2f}")

    # Calculate statistics
    successful_en = sum(1 for r in results_english if r["error"] is None)
    successful_tr = sum(1 for r in results_turkish if r["error"] is None)

    print(f"English: {successful_en}/{len(results_english)} successful")
    print(f"Turkish: {successful_tr}/{len(results_turkish)} successful")

    return {
        "english": {
            "metadata": {
                "model": model["name"],
                "model_id": model["id"],
                "provider": model["provider"],
                "turkish_support": model["turkish_support"],
                "strategy": STRATEGY,
                "source_language": source_language,
                "prompt_language": "english",
                "is_authentic_language": (source_language == "english"),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_samples": len(results_english),
                "rate_limit_delay": model["rate_limit_delay"],
                "experimental_design": [
                    f"This file: {source_language}-sourced content, English prompts",
                    "Within-subjects: Each tweet tested in both languages",
                    "Controlled content: Same tweet in TR and EN",
                ],
            },
            "results": results_english,
            "statistics": {
                "total": len(results_english),
                "successful": successful_en,
                "failed": len(results_english) - successful_en,
            },
        },
        "turkish": {
            "metadata": {
                "model": model["name"],
                "model_id": model["id"],
                "provider": model["provider"],
                "turkish_support": model["turkish_support"],
                "strategy": STRATEGY,
                "source_language": source_language,
                "prompt_language": "turkish",
                "is_authentic_language": (source_language == "turkish"),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_samples": len(results_turkish),
                "rate_limit_delay": model["rate_limit_delay"],
                "experimental_design": [
                    f"This file: {source_language}-sourced content, Turkish prompts",
                    "Within-subjects: Each tweet tested in both languages",
                    "Controlled content: Same tweet in TR and EN",
                ],
            },
            "results": results_turkish,
            "statistics": {
                "total": len(results_turkish),
                "successful": successful_tr,
                "failed": len(results_turkish) - successful_tr,
            },
        },
        "total_cost": total_cost,
    }


def save_results(data: Dict, model_name: str, source_language: str, prompt_language: str):
    """Save results to JSON file."""
    model_slug = model_name.lower().replace(" ", "-").replace("/", "-")
    timestamp = data["metadata"]["timestamp"]
    filename = f"{model_slug}_{source_language}-sourced_{prompt_language}_{timestamp}.json"

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main execution."""
    response = input("Proceed with generation? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Load both sets of translation pairs
    try:

        turkish_sourced_pairs = load_translation_pairs(
            TURKISH_SOURCED_FILE,
            SAMPLES_PER_SOURCE,
            SEED_TURKISH_SOURCED,
            "turkish"
        )

        english_sourced_pairs = load_translation_pairs(
            ENGLISH_SOURCED_FILE,
            SAMPLES_PER_SOURCE,
            SEED_ENGLISH_SOURCED,
            "english"
        )

        print(f"Total loaded: {len(turkish_sourced_pairs) + len(english_sourced_pairs)} pairs")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Generate responses for each model
    grand_total_cost = 0.0

    for model_idx, model in enumerate(MODELS):

        model_total_cost = 0.0

        # CONDITION 1: TURKISH-SOURCED CONTENT
        turkish_sourced_data = generate_responses_for_condition(
            model,
            turkish_sourced_pairs,
            "turkish"
        )

        save_results(turkish_sourced_data["english"], model["name"], "turkish", "english")
        save_results(turkish_sourced_data["turkish"], model["name"], "turkish", "turkish")

        model_total_cost += turkish_sourced_data["total_cost"]

        # CONDITION 2: ENGLISH-SOURCED CONTENT

        english_sourced_data = generate_responses_for_condition(
            model,
            english_sourced_pairs,
            "english"
        )

        save_results(english_sourced_data["english"], model["name"], "english", "english")
        save_results(english_sourced_data["turkish"], model["name"], "english", "turkish")

        model_total_cost += english_sourced_data["total_cost"]

        grand_total_cost += model_total_cost


if __name__ == "__main__":
    main()
