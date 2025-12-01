"""Analysis utilities for comparing results"""

from typing import Dict, List, Tuple
import json
from pathlib import Path


def calculate_gap(
    turkish_results: List[Dict],
    english_results: List[Dict],
    metric: str = 'harmful_rate'
) -> Dict:
    """
    Calculate cross lingual gap in harmful content production.

    Args:
        turkish_results: List of Turkish classification results
        english_results: List of English classification results
        metric: Which metric to calculate gap for

    Returns:
        Dict with gap statistics
    """
    # Count harmful content
    tr_harmful = sum(1 for r in turkish_results if r.get('is_harmful', False))
    en_harmful = sum(1 for r in english_results if r.get('is_harmful', False))

    tr_total = len(turkish_results)
    en_total = len(english_results)

    tr_rate = tr_harmful / tr_total if tr_total > 0 else 0
    en_rate = en_harmful / en_total if en_total > 0 else 0

    gap = tr_rate - en_rate
    gap_pp = gap * 100  # Percentage points

    return {
        'turkish_harmful': tr_harmful,
        'turkish_total': tr_total,
        'turkish_rate': tr_rate,
        'turkish_rate_pct': tr_rate * 100,
        'english_harmful': en_harmful,
        'english_total': en_total,
        'english_rate': en_rate,
        'english_rate_pct': en_rate * 100,
        'gap': gap,
        'gap_pp': gap_pp,
        'gap_direction': 'TR>EN' if gap > 0 else 'EN>TR' if gap < 0 else 'EQUAL'
    }


def analyze_results(
    results_dir: Path,
    model_name: str,
    strategy_name: str
) -> Dict:
    """
    Analyze results for a specific (model, strategy) combination.

    Args:
        results_dir: Directory containing result files
        model_name: Name of the model
        strategy_name: Name of the strategy

    Returns:
        Dict with analysis results
    """
    # Load Turkish results
    tr_file = results_dir / f"{model_name}_{strategy_name}_turkish.json"
    en_file = results_dir / f"{model_name}_{strategy_name}_english.json"

    if not tr_file.exists() or not en_file.exists():
        return {'error': f'Results not found for {model_name}/{strategy_name}'}

    with open(tr_file, 'r') as f:
        tr_data = json.load(f)

    with open(en_file, 'r') as f:
        en_data = json.load(f)

    # Extract results
    tr_results = tr_data.get('results', [])
    en_results = en_data.get('results', [])

    # Calculate gap
    gap_stats = calculate_gap(tr_results, en_results)

    return {
        'model': model_name,
        'strategy': strategy_name,
        **gap_stats
    }


def compare_models(
    results_dir: Path,
    models: List[str],
    strategy: str
) -> List[Dict]:
    """
    Compare multiple models on the same strategy.

    Args:
        results_dir: Directory containing results
        models: List of model names
        strategy: Strategy name

    Returns:
        List of comparison results sorted by gap
    """
    comparisons = []

    for model in models:
        result = analyze_results(results_dir, model, strategy)
        if 'error' not in result:
            comparisons.append(result)

    # Sort by gap (largest first)
    comparisons.sort(key=lambda x: x['gap_pp'], reverse=True)

    return comparisons


def print_comparison_table(comparisons: List[Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-LINGUAL GAP COMPARISON")
    print("=" * 100)
    print(f"{'Model':<40} {'Strategy':<30} {'TR%':>8} {'EN%':>8} {'Gap':>8}")
    print("-" * 100)

    for comp in comparisons:
        print(
            f"{comp['model']:<40} "
            f"{comp['strategy']:<30} "
            f"{comp['turkish_rate_pct']:>7.1f}% "
            f"{comp['english_rate_pct']:>7.1f}% "
            f"{comp['gap_pp']:>+7.1f}pp"
        )

    print("=" * 100 + "\n")
