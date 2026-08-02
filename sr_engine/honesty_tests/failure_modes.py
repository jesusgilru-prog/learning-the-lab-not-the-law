"""Test 1: Failure-mode characterization.

Runs the SR pipeline on SRSD equations that are NOT power-laws.
Verifies the method fails gracefully and doesn't produce false positives.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data', 'processed', 'loaders'))

from sr_engine.pipeline import audit_law
from srsd_feynman import load_srsd_feynman


def identify_non_powerlaw_equations():
    """Find equations NOT in the power-law subset.

    Returns
    -------
    list of dict
        Each dict has keys: difficulty, equation_idx, equation_id.
    """
    subset_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'srsd_powerlaw_subset.json')
    with open(subset_path) as f:
        subset = json.load(f)

    powerlaw_ids = {c['equation_id'] for c in subset['candidates']}

    non_powerlaw = []
    for diff, n_eqs in [('easy', 30), ('medium', 40), ('hard', 50)]:
        for idx in range(n_eqs):
            eq_id = f"feynman_{diff}_{idx:03d}"
            if eq_id not in powerlaw_ids:
                non_powerlaw.append({
                    'difficulty': diff,
                    'equation_idx': idx,
                    'equation_id': eq_id,
                })

    return non_powerlaw


def run_failure_mode_test(max_problems: int = 47, seed: int = 42):
    """Run the SR pipeline on non-power-law equations.

    Parameters
    ----------
    max_problems : int
        Maximum number of non-power-law problems to test.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Report with results and statistics.
    """
    rng = np.random.default_rng(seed)
    non_pl = identify_non_powerlaw_equations()

    n_test = min(max_problems, len(non_pl))
    if n_test < len(non_pl):
        indices = rng.choice(len(non_pl), size=n_test, replace=False)
        test_cases = [non_pl[i] for i in sorted(indices)]
    else:
        test_cases = non_pl

    results = []
    times = []

    for case in test_cases:
        try:
            eq = load_srsd_feynman(
                split='train',
                difficulty=case['difficulty'],
                equation_idx=case['equation_idx'],
            )
        except Exception as e:
            results.append({
                'equation_id': case['equation_id'],
                'difficulty': case['difficulty'],
                'status': 'load_error',
                'error': str(e),
            })
            continue

        t0 = time.time()
        result = audit_law(
            X=eq['X'],
            y=eq['y'],
            feature_names=eq['feature_names'],
            equation_id=case['equation_id'],
            n_bootstrap=100,
            seed=seed,
        )
        dt = time.time() - t0
        times.append(dt)

        results.append({
            'equation_id': case['equation_id'],
            'difficulty': case['difficulty'],
            'r2_logspace': result.r2_logspace,
            'r2_original': result.r2_original,
            'equation_str': result.equation_str,
            'best_method': result.best_method,
            'complexity': result.complexity,
            'n_features': eq['n_features'],
            'time_s': dt,
            'status': 'ok',
        })

    # Classify results
    ok_results = [r for r in results if r['status'] == 'ok']
    n_rejected = sum(1 for r in ok_results if r['r2_logspace'] < 0.5)
    n_low_r2 = sum(1 for r in ok_results if 0.5 <= r['r2_logspace'] < 0.9)
    n_high_r2 = sum(1 for r in ok_results if r['r2_logspace'] >= 0.9)
    # "False positives" — high R² on non-power-law equations.
    # These are methodological concerns: the model fits well in log-space
    # but the underlying equation is NOT a power law.
    n_false_positive_suspect = sum(
        1 for r in ok_results if r['r2_logspace'] >= 0.95
    )

    report = {
        'description': 'Failure-mode characterization on non-power-law SRSD equations',
        'total_non_powerlaw': len(non_pl),
        'tested': n_test,
        'load_errors': sum(1 for r in results if r['status'] == 'load_error'),
        'rejected_r2_lt_05': n_rejected,
        'marginal_r2_05_09': n_low_r2,
        'high_r2_ge_09': n_high_r2,
        'false_positive_suspect_r2_ge_095': n_false_positive_suspect,
        'avg_r2_logspace': float(np.mean([r['r2_logspace'] for r in ok_results])) if ok_results else None,
        'avg_time_s': float(np.mean(times)) if times else None,
        'results': results,
    }

    return report


def plot_failure_modes(report: dict, output_path: str):
    """Generate scatter plot: R² vs equation index, colored by category.

    Parameters
    ----------
    report : dict
        Output from run_failure_mode_test.
    output_path : str
        Path to save PNG figure.
    """
    ok_results = [r for r in report['results'] if r['status'] == 'ok']
    if not ok_results:
        return

    r2_values = [r['r2_logspace'] for r in ok_results]
    difficulties = [r['difficulty'] for r in ok_results]

    color_map = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    colors = [color_map.get(d, '#95a5a6') for d in difficulties]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(range(len(r2_values)), r2_values, c=colors, alpha=0.7, s=50, edgecolors='k', linewidths=0.5)

    # Threshold lines
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Rejection threshold (R²=0.5)')
    ax.axhline(y=0.9, color='orange', linestyle='--', linewidth=1, label='Concern threshold (R²=0.9)')
    ax.axhline(y=0.95, color='darkred', linestyle='--', linewidth=1, label='False positive threshold (R²=0.95)')

    ax.set_xlabel('Non-power-law equation index', fontsize=12)
    ax.set_ylabel('R² in log-space', fontsize=12)
    ax.set_title('Failure-Mode Characterization: SR Pipeline on Non-Power-Law Equations', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    # Custom legend for difficulty
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Easy'),
        Patch(facecolor='#f39c12', label='Medium'),
        Patch(facecolor='#e74c3c', label='Hard'),
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=legend_elements, loc='upper left', title='Difficulty', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {output_path}")


def main():
    print("=" * 60)
    print("TEST 1: Failure-Mode Characterization")
    print("=" * 60)

    report = run_failure_mode_test(max_problems=47, seed=42)

    # Save report
    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'failure_modes_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")

    # Print summary
    print(f"\nTotal non-power-law equations: {report['total_non_powerlaw']}")
    print(f"Tested: {report['tested']}")
    print(f"Load errors: {report['load_errors']}")
    print(f"Rejected (R² < 0.5): {report['rejected_r2_lt_05']}")
    print(f"Marginal (0.5 ≤ R² < 0.9): {report['marginal_r2_05_09']}")
    print(f"High R² (≥ 0.9): {report['high_r2_ge_09']}")
    print(f"False positive suspects (R² ≥ 0.95): {report['false_positive_suspect_r2_ge_095']}")
    print(f"Avg R² log-space: {report['avg_r2_logspace']:.4f}" if report['avg_r2_logspace'] else "")
    print(f"Avg time: {report['avg_time_s']:.2f}s" if report['avg_time_s'] else "")

    # Plot
    fig_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'failure_modes.png')
    plot_failure_modes(report, fig_path)

    return report


if __name__ == '__main__':
    main()
