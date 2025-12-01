#!/usr/bin/env python3
"""
Generate tables from analysis data.
"""

import json
from pathlib import Path

# Paths
ANALYSIS_JSON = Path(__file__).parent.parent.parent / "multi_model_analysis_date_time.json"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load the analysis JSON data."""
    with open(ANALYSIS_JSON) as f:
        return json.load(f)


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< .001"
    elif p < 0.01:
        return f"< .01"
    elif p < 0.05:
        return f"< .05"
    else:
        return f".{int(p*1000):03d}"


def format_significance(sig: str) -> str:
    """Convert significance to stars."""
    if "p < 0.01" in sig:
        return "**"
    elif "p < 0.05" in sig:
        return "*"
    else:
        return ""


def generate_main_results_table(data: dict, classifier: str = "gemma") -> str:
    """
    Shows refusal rates for Turkish vs English prompts across all models.
    """
    output = []
    output.append("=" * 100)
    output.append("Cross Lingual Safety Alignment Gaps (Marginal Comparison)")
    output.append(f"Classifier: {'GemmaGuard' if classifier == 'gemma' else classifier.title()}")
    output.append("=" * 100)
    output.append("")
    output.append(f"{'Model':<20} {'Provider':<20} {'TR Rate':>8} {'EN Rate':>8} {'Gap (pp)':>10} {'p-value':>10} {'Cohen h':>10} {'Sig':>5}")
    output.append("-" * 100)

    for model in data["models"]:
        clf_data = model["classifiers"][classifier]
        marginal = clf_data["marginal_comparison"]

        tr_rate = marginal["turkish_prompt_rate"]
        en_rate = marginal["english_prompt_rate"]
        gap = marginal["gap_percentage_points"]
        p_val = marginal["p_value"]
        cohens_h = marginal["cohens_h"]
        sig = format_significance(marginal["significance"])

        output.append(
            f"{model['model_name']:<20} {model['provider']:<20} "
            f"{tr_rate*100:>7.1f}% {en_rate*100:>7.1f}% "
            f"{gap:>+9.1f} {format_p_value(p_val):>10} {cohens_h:>+10.3f} {sig:>5}"
        )

    return "\n".join(output)


def generate_native_parity_table(data: dict, classifier: str = "gemma") -> str:
    """
    Compares TR→TR vs EN→EN (same source and prompt language).
    """
    output = []
    output.append("=" * 100)
    output.append("Native Language Parity (Pure Native Comparison)")
    output.append(f"Classifier: {'GemmaGuard' if classifier == 'gemma' else classifier.title()}")
    output.append("=" * 100)
    output.append("")
    output.append(f"{'Model':<20} {'TR→TR':>8} {'EN→EN':>8} {'Gap (pp)':>10} {'Direction':>10} {'p-value':>10} {'Cohen h':>10} {'Sig':>5}")
    output.append("-" * 100)

    parity_achieved = []

    for model in data["models"]:
        clf_data = model["classifiers"][classifier]
        native = clf_data["pure_native_comparison"]
        conditions = clf_data["conditions"]

        tr_tr = conditions["tr_sourced_tr_prompt"]["rate"]
        en_en = conditions["en_sourced_en_prompt"]["rate"]
        gap = native["gap_percentage_points"]
        direction = native["direction"]
        p_val = native["p_value"]
        cohens_h = native["cohens_h"]
        sig = format_significance(native["significance"])

        if sig == "":
            parity_achieved.append(model["model_name"])

        output.append(
            f"{model['model_name']:<20} "
            f"{tr_tr*100:>7.1f}% {en_en*100:>7.1f}% "
            f"{gap:>+9.1f} {direction:>10} {format_p_value(p_val):>10} {cohens_h:>+10.3f} {sig:>5}"
        )

    return "\n".join(output)


def generate_translation_effects_table(data: dict, classifier: str = "gemma") -> str:
    """
    Shows within-subjects effects of translation.
    """
    output = []
    output.append("=" * 100)
    output.append("Translation Direction Effects (Within-Subjects)")
    output.append(f"Classifier: {'GemmaGuard' if classifier == 'gemma' else classifier.title()}")
    output.append("=" * 100)
    output.append("")
    output.append(f"{'Model':<20} {'EN→EN':>7} {'EN→TR':>7} {'Effect':>8} {'Sig':>4} | {'TR→TR':>7} {'TR→EN':>7} {'Effect':>8} {'Sig':>4}")
    output.append("-" * 100)

    for model in data["models"]:
        clf_data = model["classifiers"][classifier]
        conditions = clf_data["conditions"]
        en_within = clf_data["within_subjects_english_source"]
        tr_within = clf_data["within_subjects_turkish_source"]

        en_en = conditions["en_sourced_en_prompt"]["rate"] * 100
        en_tr = conditions["en_sourced_tr_prompt"]["rate"] * 100
        tr_tr = conditions["tr_sourced_tr_prompt"]["rate"] * 100
        tr_en = conditions["tr_sourced_en_prompt"]["rate"] * 100

        en_effect = en_within["gap_percentage_points"]
        tr_effect = tr_within["gap_percentage_points"]

        en_sig = format_significance(en_within["significance"])
        tr_sig = format_significance(tr_within["significance"])

        output.append(
            f"{model['model_name']:<20} "
            f"{en_en:>6.1f}% {en_tr:>6.1f}% {en_effect:>+7.1f} {en_sig:>4} | "
            f"{tr_tr:>6.1f}% {tr_en:>6.1f}% {tr_effect:>+7.1f} {tr_sig:>4}"
        )

    return "\n".join(output)


def generate_full_conditions_table(data: dict, classifier: str = "gemma") -> str:
    """
    Shows all four conditions for each model.
    """
    output = []
    output.append("=" * 100)
    output.append("2×2 Factorial Design Results")
    output.append(f"Classifier: {'GemmaGuard' if classifier == 'gemma' else classifier.title()}")
    output.append("=" * 100)
    output.append("")
    output.append(f"{'Model':<20} {'TR→TR':>10} {'TR→EN':>10} {'EN→EN':>10} {'EN→TR':>10}")
    output.append("-" * 100)

    for model in data["models"]:
        clf_data = model["classifiers"][classifier]
        conditions = clf_data["conditions"]

        tr_tr = conditions["tr_sourced_tr_prompt"]["rate"] * 100
        tr_en = conditions["tr_sourced_en_prompt"]["rate"] * 100
        en_en = conditions["en_sourced_en_prompt"]["rate"] * 100
        en_tr = conditions["en_sourced_tr_prompt"]["rate"] * 100

        output.append(
            f"{model['model_name']:<20} "
            f"{tr_tr:>9.1f}% {tr_en:>9.1f}% {en_en:>9.1f}% {en_tr:>9.1f}%"
        )

    return "\n".join(output)


def main():
    """Generate all tables."""
    data = load_data()

    print(f"Found {data['n_models']} models")

    # Generate tables for GemmaGuard
    classifier = "gemma"

    tables = []
    tables.append(generate_main_results_table(data, classifier))
    tables.append(generate_native_parity_table(data, classifier))
    tables.append(generate_translation_effects_table(data, classifier))
    tables.append(generate_full_conditions_table(data, classifier))

    # Write to output file
    output_file = OUTPUT_DIR / "tables.txt"
    with open(output_file, "w") as f:
        f.write("\n\n".join(tables))

    print(f"\nTables written to: {output_file}")


if __name__ == "__main__":
    main()
