#!/usr/bin/env python3
"""
Multi-Model Cross Lingual Safety Analysis
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats

# Configuration
WILDGUARD_DIR = Path("wildguard_classified")
GEMMA_DIR = Path("gemma_classified")
LOGREG_DIR = Path("logreg_classified")
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"id": "gpt-5-nano", "name": "GPT-5-nano", "provider": "OpenAI", "turkish": "Official"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "turkish": "Official"},
    {"id": "gpt-4o-mini", "name": "GPT-4o-mini", "provider": "OpenAI", "turkish": "Official"},
    {"id": "mixtral-8x22b", "name": "Mixtral 8x22B", "provider": "Mistral (France)", "turkish": "Unofficial"},
    {"id": "deepseek-v3", "name": "DeepSeek V3", "provider": "DeepSeek (China)", "turkish": "Unofficial"},
    {"id": "qwen2.5-72b", "name": "Qwen2.5 72B", "provider": "Alibaba (China)", "turkish": "Unofficial"},
    {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "provider": "Anthropic (USA)", "turkish": "Unofficial"},
    {"id": "llama-3.3-70b", "name": "Llama 3.3 70B", "provider": "Meta (USA)", "turkish": "Unofficial"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Google (USA)", "turkish": "Official"},
    {"id": "deepseek-v3.1", "name": "DeepSeek V3.1", "provider": "DeepSeek (China)", "turkish": "Unofficial"},
]


def load_file(directory: Path, model_id: str, source_lang: str, prompt_lang: str, suffix: str) -> Dict:
    """Load classified file for specific condition."""
    pattern = f"{model_id}_{source_lang}-sourced_{prompt_lang}_*{suffix}*.json"
    files = list(directory.glob(pattern))

    if not files:
        return None

    if len(files) > 1:
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    with open(files[0]) as f:
        return json.load(f)


def calculate_refusal_rate(data: Dict, classifier: str) -> Tuple[int, int, float]:
    """
    Calculate refusal rate for a classifier.
    """
    if data is None:
        return 0, 0, 0.0

    results = data["results"]
    n_total = len(results)

    if classifier == "wildguard":
        refusals = sum(1 for r in results if r.get("wildguard_refusal", False))
    elif classifier == "gemma":
        refusals = sum(1 for r in results if r.get("is_refusal", False))
    elif classifier == "logreg":
        refusals = sum(1 for r in results if r.get("logreg_classification", {}).get("logreg_is_refusal", False))
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    rate = refusals / n_total if n_total > 0 else 0.0
    return refusals, n_total, rate


def fishers_exact_test(n1_refusal: int, n1_total: int, n2_refusal: int, n2_total: int) -> Tuple[float, str]:
    """
    Perform Fisher's exact test.

    """
    n1_compliance = n1_total - n1_refusal
    n2_compliance = n2_total - n2_refusal

    contingency_table = [[n1_refusal, n1_compliance],
                         [n2_refusal, n2_compliance]]

    _, p_value = stats.fisher_exact(contingency_table)

    if p_value < 0.05:
        sig = "p < 0.05"
    elif p_value < 0.10:
        sig = "p < 0.10"
    else:
        sig = "n.s."

    return p_value, sig


def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def mcnemar_test_from_rates(n_refusal_cond1: int, n_refusal_cond2: int, n_total: int) -> Tuple[float, str]:
    """
    Perform McNemar's test for paired samples.
    """
    p1 = n_refusal_cond1 / n_total
    p2 = n_refusal_cond2 / n_total

    # Pooled proportion for standard error
    p_pooled = (n_refusal_cond1 + n_refusal_cond2) / (2 * n_total)
    se = np.sqrt(2 * p_pooled * (1 - p_pooled) / n_total)

    if se == 0:
        return 1.0, "n.s."

    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed

    if p_value < 0.01:
        sig = "p < 0.01"
    elif p_value < 0.05:
        sig = "p < 0.05"
    else:
        sig = "n.s."

    return p_value, sig


def analyze_model(model: Dict) -> Dict:
    """Analyze one model across all conditions and classifiers."""
    model_id = model["id"]
    model_name = model["name"]

    results = {
        "model_id": model_id,
        "model_name": model_name,
        "provider": model["provider"],
        "turkish_support": model["turkish"],
        "classifiers": {}
    }

    # Analyze each classifier
    for classifier, directory, suffix in [
        ("gemma", GEMMA_DIR, "gemma"),
        ("wildguard", WILDGUARD_DIR, "wildguard"),
        ("logreg", LOGREG_DIR, "logreg"),
    ]:
        # Load all 4 conditions
        tr_tr = load_file(directory, model_id, "turkish", "turkish", suffix)
        tr_en = load_file(directory, model_id, "turkish", "english", suffix)
        en_en = load_file(directory, model_id, "english", "english", suffix)
        en_tr = load_file(directory, model_id, "english", "turkish", suffix)

        # Calculate refusal rates
        tr_tr_ref, tr_tr_n, tr_tr_rate = calculate_refusal_rate(tr_tr, classifier)
        tr_en_ref, tr_en_n, tr_en_rate = calculate_refusal_rate(tr_en, classifier)
        en_en_ref, en_en_n, en_en_rate = calculate_refusal_rate(en_en, classifier)
        en_tr_ref, en_tr_n, en_tr_rate = calculate_refusal_rate(en_tr, classifier)

        # Pure native comparison (TR-sourced TR vs EN-sourced EN)
        gap_pp = (tr_tr_rate - en_en_rate) * 100
        p_value, sig = fishers_exact_test(tr_tr_ref, tr_tr_n, en_en_ref, en_en_n)
        effect_size = cohens_h(tr_tr_rate, en_en_rate)

        # Within-subjects: English source (EN→EN vs EN→TR)
        en_within_gap = (en_en_rate - en_tr_rate) * 100
        en_within_p, en_within_sig = mcnemar_test_from_rates(en_en_ref, en_tr_ref, en_en_n)
        en_within_h = cohens_h(en_en_rate, en_tr_rate)

        # Within-subjects: Turkish source (TR→TR vs TR→EN)
        tr_within_gap = (tr_tr_rate - tr_en_rate) * 100
        tr_within_p, tr_within_sig = mcnemar_test_from_rates(tr_tr_ref, tr_en_ref, tr_tr_n)
        tr_within_h = cohens_h(tr_tr_rate, tr_en_rate)

        # Calculate marginal rates (collapsed across source)
        total_turkish_prompt = tr_tr_ref + en_tr_ref
        total_turkish_n = tr_tr_n + en_tr_n
        turkish_marginal = total_turkish_prompt / total_turkish_n if total_turkish_n > 0 else 0.0

        total_english_prompt = tr_en_ref + en_en_ref
        total_english_n = tr_en_n + en_en_n
        english_marginal = total_english_prompt / total_english_n if total_english_n > 0 else 0.0

        marginal_gap = (turkish_marginal - english_marginal) * 100

        # Marginal comparison (Turkish prompts vs English prompts, collapsed across source)
        marginal_p, marginal_sig = fishers_exact_test(
            total_turkish_prompt, total_turkish_n,
            total_english_prompt, total_english_n
        )
        marginal_h = cohens_h(turkish_marginal, english_marginal)

        results["classifiers"][classifier] = {
            "conditions": {
                "tr_sourced_tr_prompt": {"refusals": tr_tr_ref, "total": tr_tr_n, "rate": tr_tr_rate},
                "tr_sourced_en_prompt": {"refusals": tr_en_ref, "total": tr_en_n, "rate": tr_en_rate},
                "en_sourced_en_prompt": {"refusals": en_en_ref, "total": en_en_n, "rate": en_en_rate},
                "en_sourced_tr_prompt": {"refusals": en_tr_ref, "total": en_tr_n, "rate": en_tr_rate},
            },
            "pure_native_comparison": {
                "turkish_rate": tr_tr_rate,
                "english_rate": en_en_rate,
                "gap_percentage_points": gap_pp,
                "direction": "TR > EN" if gap_pp > 0 else "EN > TR" if gap_pp < 0 else "No gap",
                "p_value": p_value,
                "significance": sig,
                "cohens_h": effect_size,
            },
            "within_subjects_english_source": {
                "en_to_en": en_en_rate,
                "en_to_tr": en_tr_rate,
                "gap_percentage_points": en_within_gap,
                "direction": "EN>TR" if en_within_gap > 0 else "TR>EN" if en_within_gap < 0 else "No gap",
                "p_value": en_within_p,
                "significance": en_within_sig,
                "cohens_h": en_within_h,
            },
            "within_subjects_turkish_source": {
                "tr_to_tr": tr_tr_rate,
                "tr_to_en": tr_en_rate,
                "gap_percentage_points": tr_within_gap,
                "direction": "TR>EN" if tr_within_gap > 0 else "EN>TR" if tr_within_gap < 0 else "No gap",
                "p_value": tr_within_p,
                "significance": tr_within_sig,
                "cohens_h": tr_within_h,
            },
            "marginal_comparison": {
                "turkish_prompt_rate": turkish_marginal,
                "english_prompt_rate": english_marginal,
                "gap_percentage_points": marginal_gap,
                "direction": "TR > EN" if marginal_gap > 0 else "EN > TR" if marginal_gap < 0 else "No gap",
                "p_value": marginal_p,
                "significance": marginal_sig,
                "cohens_h": marginal_h,
            }
        }

    return results


def main():
    """Run comprehensive multi-model analysis."""

    # Analyze all models
    all_results = []
    for model in MODELS:
        print(f"\nAnalyzing {model['name']}")
        results = analyze_model(model)
        all_results.append(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = ANALYSIS_DIR / f"multi_model_analysis_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "n_models": len(MODELS),
            "models": all_results,
        }, f, indent=2)

if __name__ == "__main__":
    main()
