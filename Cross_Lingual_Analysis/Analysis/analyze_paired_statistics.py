#!/usr/bin/env python3
"""
Compute McNemar's test and Cohen's g for paired within-subjects comparisons.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Tuple

# Paths
CLASSIFIED_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent


def load_paired_data():
    """
    Load classified data and organize by model, source, and sample_id.

    Returns dict: model_id -> source_lang -> sample_id -> {en_prompt: bool, tr_prompt: bool}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for json_file in CLASSIFIED_DIR.glob("*.json"):
        if json_file.name.startswith("summary"):
            continue

        with open(json_file) as f:
            file_data = json.load(f)

        metadata = file_data["metadata"]
        model_id = metadata["model_id"]
        source = metadata["source_language"]
        prompt = metadata["prompt_language"]

        for result in file_data["results"]:
            sample_id = result["sample_id"]
            # Use gemma classifier
            is_refusal = result.get("gemma_refusal", {}).get("prediction") == "refusal"

            # Store by prompt language
            data[model_id][source][sample_id][prompt] = is_refusal

    return data


def mcnemar_test(b: int, c: int) -> Tuple[float, float, str]:
    """
    Compute McNemar's test statistic and p-value.

    Args:
        b: discordant pair count (condition1=refusal, condition2=compliance)
        c: discordant pair count (condition1=compliance, condition2=refusal)

    Returns:
        (chi_squared, p_value, significance)
    """
    from scipy import stats

    if b + c == 0:
        return 0.0, 1.0, "n.s."

    # McNemar's chi-squared (with continuity correction)
    chi_sq = ((abs(b - c) - 1) ** 2) / (b + c)

    # p-value from chi-squared distribution with 1 df
    p_value = 1 - stats.chi2.cdf(chi_sq, 1)

    if p_value < 0.01:
        sig = "p < 0.01 **"
    elif p_value < 0.05:
        sig = "p < 0.05 *"
    else:
        sig = "n.s."

    return chi_sq, p_value, sig


def cohens_g(b: int, c: int) -> float:
    """
    Compute Cohen's g effect size for paired proportions.

    Cohen's g = P(condition2 > condition1) - 0.5
             = c / (b + c) - 0.5

    """
    if b + c == 0:
        return 0.0

    return (c / (b + c)) - 0.5


def analyze_within_subjects(data: dict, model_id: str, source: str) -> dict:
    """
    Analyze within-subjects comparison for a model and source language.

    Compares English prompt vs Turkish prompt for the same content.
    """
    samples = data[model_id][source]

    # Build contingency table
    # a: both refusal, b: EN refusal/TR compliance, c: EN compliance/TR refusal, d: both compliance
    a = b = c = d = 0

    for sample_id, prompts in samples.items():
        if "english" not in prompts or "turkish" not in prompts:
            continue

        en_refusal = prompts["english"]
        tr_refusal = prompts["turkish"]

        if en_refusal and tr_refusal:
            a += 1
        elif en_refusal and not tr_refusal:
            b += 1  # EN refusal, TR compliance
        elif not en_refusal and tr_refusal:
            c += 1  # EN compliance, TR refusal
        else:
            d += 1

    n_total = a + b + c + d

    # Compute statistics
    chi_sq, p_value, sig = mcnemar_test(b, c)
    g = cohens_g(b, c)

    # Rates
    en_rate = (a + b) / n_total if n_total > 0 else 0
    tr_rate = (a + c) / n_total if n_total > 0 else 0
    gap = (tr_rate - en_rate) * 100

    return {
        "n_total": n_total,
        "concordant_refusal": a,
        "concordant_compliance": d,
        "discordant_b": b,  # EN refusal, TR compliance
        "discordant_c": c,  # EN compliance, TR refusal
        "en_rate": en_rate,
        "tr_rate": tr_rate,
        "gap_pp": gap,
        "mcnemar_chi_sq": chi_sq,
        "p_value": p_value,
        "significance": sig,
        "cohens_g": g,
    }


def generate_paired_statistics_table(data: dict) -> str:
    """Generate table with McNemar's test and Cohen's g for all models."""
    output = []
    output.append("Classifier: GemmaGuard")
    output.append("")

    model_order = [
        "gpt-5-nano", "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
        "mistralai/mixtral-8x22b-instruct", "deepseek/deepseek-chat-v3-0324",
        "qwen/qwen-2.5-72b-instruct", "anthropic/claude-sonnet-4.5",
        "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat-v3.1"
    ]

    model_names = {
        "gpt-5-nano": "GPT-5-nano",
        "gpt-4o-2024-05-13": "GPT-4o",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
        "qwen/qwen-2.5-72b-instruct": "Qwen2.5 72B",
        "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
        "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    }

    # English-sourced content (EN→EN vs EN→TR)
    output.append("ENGLISH-SOURCED CONTENT: EN→EN vs EN→TR (Effect of translating to Turkish)")
    output.append("-" * 140)
    output.append(f"{'Model':<20} {'EN→EN':>7} {'EN→TR':>7} {'Gap':>8} {'b':>5} {'c':>5} {'χ²':>8} {'p':>10} {'Cohen g':>10} {'Sig':>12}")
    output.append("-" * 140)

    for model_id in model_order:
        if model_id not in data:
            continue

        stats = analyze_within_subjects(data, model_id, "english")
        name = model_names.get(model_id, model_id)

        output.append(
            f"{name:<20} "
            f"{stats['en_rate']*100:>6.1f}% {stats['tr_rate']*100:>6.1f}% "
            f"{stats['gap_pp']:>+7.1f} "
            f"{stats['discordant_b']:>5} {stats['discordant_c']:>5} "
            f"{stats['mcnemar_chi_sq']:>8.2f} {stats['p_value']:>10.4f} "
            f"{stats['cohens_g']:>+10.3f} {stats['significance']:>12}"
        )

    output.append("-" * 140)
    output.append("")

    # Turkish-sourced content (TR→TR vs TR→EN)
    output.append("TURKISH-SOURCED CONTENT: TR→TR vs TR→EN (Effect of translating to English)")
    output.append("-" * 140)
    output.append(f"{'Model':<20} {'TR→TR':>7} {'TR→EN':>7} {'Gap':>8} {'b':>5} {'c':>5} {'χ²':>8} {'p':>10} {'Cohen g':>10} {'Sig':>12}")
    output.append("-" * 140)

    for model_id in model_order:
        if model_id not in data:
            continue

        stats = analyze_within_subjects(data, model_id, "turkish")
        name = model_names.get(model_id, model_id)

        output.append(
            f"{name:<20} "
            f"{stats['en_rate']*100:>6.1f}% {stats['tr_rate']*100:>6.1f}% "
            f"{stats['gap_pp']:>+7.1f} "
            f"{stats['discordant_b']:>5} {stats['discordant_c']:>5} "
            f"{stats['mcnemar_chi_sq']:>8.2f} {stats['p_value']:>10.4f} "
            f"{stats['cohens_g']:>+10.3f} {stats['significance']:>12}"
        )

    output.append("-" * 140)
    output.append("")

    output.append("Notes:")
    output.append("- b = pairs where English prompt refused, Turkish prompt complied")
    output.append("- c = pairs where English prompt complied, Turkish prompt refused")
    output.append("- χ² = McNemar's chi-squared with continuity correction")
    output.append("- Cohen's g: small = 0.05, medium = 0.15, large = 0.25")
    output.append("- Positive g = Turkish prompt more likely to refuse")
    output.append("- * p < .05, ** p < .01")
    output.append("")

    return "\n".join(output)


def generate_detailed_summary(data: dict) -> str:
    """Generate detailed per-model summary with all paired statistics."""
    output = []
    output.append("=" * 100)
    output.append("DETAILED PAIRED STATISTICS SUMMARY")
    output.append("=" * 100)
    output.append("")

    model_order = [
        "gpt-5-nano", "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18",
        "mistralai/mixtral-8x22b-instruct", "deepseek/deepseek-chat-v3-0324",
        "qwen/qwen-2.5-72b-instruct", "anthropic/claude-sonnet-4.5",
        "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat-v3.1"
    ]

    model_names = {
        "gpt-5-nano": "GPT-5-nano",
        "gpt-4o-2024-05-13": "GPT-4o",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "mistralai/mixtral-8x22b-instruct": "Mixtral 8x22B",
        "deepseek/deepseek-chat-v3-0324": "DeepSeek V3",
        "qwen/qwen-2.5-72b-instruct": "Qwen2.5 72B",
        "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
        "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    }

    for model_id in model_order:
        if model_id not in data:
            continue

        name = model_names.get(model_id, model_id)
        output.append(f"\n{name}")
        output.append("-" * 60)

        # English-sourced
        en_stats = analyze_within_subjects(data, model_id, "english")
        output.append("  English-sourced (EN→EN vs EN→TR):")
        output.append(f"    Contingency: a={en_stats['concordant_refusal']}, "
                     f"b={en_stats['discordant_b']}, "
                     f"c={en_stats['discordant_c']}, "
                     f"d={en_stats['concordant_compliance']}")
        output.append(f"    Gap: {en_stats['gap_pp']:+.1f}pp "
                     f"({en_stats['en_rate']*100:.1f}% → {en_stats['tr_rate']*100:.1f}%)")
        output.append(f"    McNemar χ² = {en_stats['mcnemar_chi_sq']:.2f}, "
                     f"p = {en_stats['p_value']:.4f}")
        output.append(f"    Cohen's g = {en_stats['cohens_g']:+.3f}")

        # Turkish-sourced
        tr_stats = analyze_within_subjects(data, model_id, "turkish")
        output.append("  Turkish-sourced (TR→TR vs TR→EN):")
        output.append(f"    Contingency: a={tr_stats['concordant_refusal']}, "
                     f"b={tr_stats['discordant_b']}, "
                     f"c={tr_stats['discordant_c']}, "
                     f"d={tr_stats['concordant_compliance']}")
        output.append(f"    Gap: {tr_stats['gap_pp']:+.1f}pp "
                     f"({tr_stats['en_rate']*100:.1f}% → {tr_stats['tr_rate']*100:.1f}%)")
        output.append(f"    McNemar χ² = {tr_stats['mcnemar_chi_sq']:.2f}, "
                     f"p = {tr_stats['p_value']:.4f}")
        output.append(f"    Cohen's g = {tr_stats['cohens_g']:+.3f}")

    output.append("")
    return "\n".join(output)


def main():
    """Generate paired statistics analysis."""
    data = load_paired_data()

    # Generate tables
    table = generate_paired_statistics_table(data)
    summary = generate_detailed_summary(data)

    # Write to output file
    output_file = OUTPUT_DIR / "thesis_table_paired_statistics.txt"
    with open(output_file, "w") as f:
        f.write(table)
        f.write("\n\n")
        f.write(summary)

    print(f"\nTable written to: {output_file}")


if __name__ == "__main__":
    main()
