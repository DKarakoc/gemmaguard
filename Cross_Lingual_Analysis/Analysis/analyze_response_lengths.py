#!/usr/bin/env python3
"""
Analyze response lengths across models and conditions.
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

# Paths
CLASSIFIED_DIR = Path(__file__).parent.parent.parent / "gemma_classified"
OUTPUT_DIR = Path(__file__).parent


def load_classified_files():
    """Load all classified JSON files and organize by model/condition."""
    data = defaultdict(lambda: defaultdict(list))

    for json_file in CLASSIFIED_DIR.glob("*.json"):
        if json_file.name.startswith("summary"):
            continue

        with open(json_file) as f:
            file_data = json.load(f)

        metadata = file_data["metadata"]
        model_id = metadata["model_id"]
        source = metadata["source_language"]
        prompt = metadata["prompt_language"]

        # Map to condition code
        condition_map = {
            ("turkish", "turkish"): "TR→TR",
            ("turkish", "english"): "TR→EN",
            ("english", "english"): "EN→EN",
            ("english", "turkish"): "EN→TR",
        }
        condition = condition_map[(source, prompt)]

        for result in file_data["results"]:
            response = result.get("response", "")
            if response:
                word_count = len(response.split())
                char_count = len(response)
                token_count = result.get("tokens", {}).get("completion", 0)

                data[model_id][condition].append({
                    "word_count": word_count,
                    "char_count": char_count,
                    "token_count": token_count,
                })

    return data


def generate_length_table(data: dict) -> str:
    """Generate Table 5: Response Lengths."""
    output = []
    output.append("=" * 120)
    output.append("Average Response Lengths by Model and Condition")
    output.append("=" * 120)
    output.append("")

    # Word count table
    output.append("WORD COUNTS (mean ± std)")
    output.append("-" * 120)
    output.append(f"{'Model':<20} {'TR→TR':>20} {'TR→EN':>20} {'EN→EN':>20} {'EN→TR':>20}")
    output.append("-" * 120)

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

        model_data = data[model_id]
        row = [model_names.get(model_id, model_id)]

        for condition in ["TR→TR", "TR→EN", "EN→EN", "EN→TR"]:
            if condition in model_data:
                words = [d["word_count"] for d in model_data[condition]]
                mean = statistics.mean(words)
                std = statistics.stdev(words) if len(words) > 1 else 0
                row.append(f"{mean:.1f} ± {std:.1f}")
            else:
                row.append("N/A")

        output.append(f"{row[0]:<20} {row[1]:>20} {row[2]:>20} {row[3]:>20} {row[4]:>20}")

    output.append("-" * 120)
    output.append("")

    # Token count table
    output.append("TOKEN COUNTS (mean ± std)")
    output.append("-" * 120)
    output.append(f"{'Model':<20} {'TR→TR':>20} {'TR→EN':>20} {'EN→EN':>20} {'EN→TR':>20}")
    output.append("-" * 120)

    for model_id in model_order:
        if model_id not in data:
            continue

        model_data = data[model_id]
        row = [model_names.get(model_id, model_id)]

        for condition in ["TR→TR", "TR→EN", "EN→EN", "EN→TR"]:
            if condition in model_data:
                tokens = [d["token_count"] for d in model_data[condition]]
                mean = statistics.mean(tokens)
                std = statistics.stdev(tokens) if len(tokens) > 1 else 0
                row.append(f"{mean:.1f} ± {std:.1f}")
            else:
                row.append("N/A")

        output.append(f"{row[0]:<20} {row[1]:>20} {row[2]:>20} {row[3]:>20} {row[4]:>20}")

    output.append("-" * 120)
    output.append("")

    # Summary by language
    output.append("SUMMARY: Turkish vs English Prompt Language")
    output.append("-" * 120)
    output.append(f"{'Model':<20} {'Turkish Prompts':>20} {'English Prompts':>20} {'Difference':>15} {'Ratio':>10}")
    output.append("-" * 120)

    for model_id in model_order:
        if model_id not in data:
            continue

        model_data = data[model_id]

        # Turkish prompts: TR→TR + EN→TR
        tr_words = []
        for condition in ["TR→TR", "EN→TR"]:
            if condition in model_data:
                tr_words.extend([d["word_count"] for d in model_data[condition]])

        # English prompts: TR→EN + EN→EN
        en_words = []
        for condition in ["TR→EN", "EN→EN"]:
            if condition in model_data:
                en_words.extend([d["word_count"] for d in model_data[condition]])

        if tr_words and en_words:
            tr_mean = statistics.mean(tr_words)
            en_mean = statistics.mean(en_words)
            diff = tr_mean - en_mean
            ratio = tr_mean / en_mean if en_mean > 0 else 0

            output.append(
                f"{model_names.get(model_id, model_id):<20} "
                f"{tr_mean:>19.1f} {en_mean:>19.1f} "
                f"{diff:>+14.1f} {ratio:>9.2f}x"
            )

    output.append("-" * 120)
    output.append("")
    output.append("Notes:")
    output.append("- Word count = number of whitespace-separated tokens")
    output.append("- Token count = completion tokens from API response")
    output.append("- Ratio > 1.0 indicates longer responses in Turkish")
    output.append("")

    return "\n".join(output)


def main():
    """Generate response length analysis."""
    data = load_classified_files()

    table = generate_length_table(data)

    # Write to output file
    output_file = OUTPUT_DIR / "thesis_table_response_lengths.txt"
    with open(output_file, "w") as f:
        f.write(table)

    print(f"\nTable written to: {output_file}")


if __name__ == "__main__":
    main()
