"""Test 3: Noise robustness characterization.

Evaluates SR pipeline degradation under increasing multiplicative
Gaussian noise: y_noisy = y * (1 + ε), ε ~ N(0, σ).
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


NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 0.83]
N_REPEATS = 10


def load_stratified_sample(n_per_difficulty: int = 10, seed: int = 42):
    """Load stratified sample of power-law equations.

    Parameters
    ----------
    n_per_difficulty : int
        Number of equations per difficulty level.
    seed : int
        Random seed.

    Returns
    -------
    list of dict
        Each has keys: equation_id, difficulty, X, y, feature_names, ground_truth.
    """
    subset_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'srsd_powerlaw_subset.json')
    with open(subset_path) as f:
        subset = json.load(f)

    rng = np.random.default_rng(seed)
    samples = []

    for diff in ['easy', 'medium', 'hard']:
        candidates = [c for c in subset['candidates'] if c['difficulty'] == diff]
        n = min(n_per_difficulty, len(candidates))
        chosen = rng.choice(len(candidates), size=n, replace=False)

        for idx in chosen:
            c = candidates[idx]
            parts = c['equation_id'].split('_')
            eq_idx = int(parts[-1])
            try:
                eq = load_srsd_feynman('train', diff, eq_idx)
            except Exception:
                continue

            samples.append({
                'equation_id': c['equation_id'],
                'difficulty': diff,
                'X': eq['X'],
                'y': eq['y'],
                'feature_names': eq['feature_names'],
                'ground_truth': {
                    'exponents': c['exponents'],
                    'coefficient': c['coefficient'],
                },
            })

    return samples


def run_noise_robustness_test(n_per_difficulty: int = 10, seed: int = 42):
    """Run noise robustness test.

    Parameters
    ----------
    n_per_difficulty : int
        Equations per difficulty level.
    seed : int
        Base seed.

    Returns
    -------
    dict
        Report with degradation curves.
    """
    samples = load_stratified_sample(n_per_difficulty, seed)
    rng = np.random.default_rng(seed)

    print(f"  Loaded {len(samples)} equations ({n_per_difficulty}/difficulty)")

    # Results structure: noise_level -> list of per-equation averaged results
    all_results = {}

    for sigma in NOISE_LEVELS:
        level_results = []
        t0_level = time.time()

        for sample in samples:
            X = sample['X']
            y_clean = sample['y']
            gt = sample['ground_truth']

            neds = []
            r2s_log = []
            r2s_orig = []
            recoveries = []

            for rep in range(N_REPEATS):
                rep_seed = seed + rep * 1000 + int(sigma * 100)
                rep_rng = np.random.default_rng(rep_seed)

                if sigma > 0:
                    noise = rep_rng.normal(0, sigma, size=len(y_clean))
                    y_noisy = y_clean * (1 + noise)
                    # Ensure positivity
                    y_noisy = np.abs(y_noisy)
                    y_noisy[y_noisy < 1e-300] = 1e-300
                else:
                    y_noisy = y_clean.copy()

                result = audit_law(
                    X=X, y=y_noisy,
                    feature_names=sample['feature_names'],
                    ground_truth=gt,
                    equation_id=sample['equation_id'],
                    n_bootstrap=50,
                    seed=rep_seed,
                )

                if result.ned is not None:
                    neds.append(result.ned)
                    recoveries.append(result.is_exact_recovery)
                r2s_log.append(result.r2_logspace)
                r2s_orig.append(result.r2_original)

            level_results.append({
                'equation_id': sample['equation_id'],
                'difficulty': sample['difficulty'],
                'mean_ned': float(np.mean(neds)) if neds else None,
                'std_ned': float(np.std(neds)) if neds else None,
                'recovery_rate': float(np.mean(recoveries)) if recoveries else None,
                'mean_r2_log': float(np.mean(r2s_log)),
                'mean_r2_orig': float(np.mean(r2s_orig)),
            })

        dt_level = time.time() - t0_level

        # Aggregate
        valid_neds = [r['mean_ned'] for r in level_results if r['mean_ned'] is not None]
        valid_rates = [r['recovery_rate'] for r in level_results if r['recovery_rate'] is not None]

        all_results[str(sigma)] = {
            'sigma': sigma,
            'n_equations': len(level_results),
            'overall_recovery_rate': float(np.mean(valid_rates)) if valid_rates else None,
            'overall_mean_ned': float(np.mean(valid_neds)) if valid_neds else None,
            'overall_mean_r2_log': float(np.mean([r['mean_r2_log'] for r in level_results])),
            'time_s': dt_level,
            'per_equation': level_results,
        }

        rate_str = f"{all_results[str(sigma)]['overall_recovery_rate']:.0%}" if all_results[str(sigma)]['overall_recovery_rate'] is not None else "N/A"
        ned_str = f"{all_results[str(sigma)]['overall_mean_ned']:.4f}" if all_results[str(sigma)]['overall_mean_ned'] is not None else "N/A"
        print(f"  σ={sigma:.2f}: recovery={rate_str}, mean NED={ned_str}, time={dt_level:.1f}s")

    # Per-difficulty breakdown
    difficulty_curves = {}
    for diff in ['easy', 'medium', 'hard']:
        curve = []
        for sigma in NOISE_LEVELS:
            level_data = all_results[str(sigma)]
            diff_eqs = [r for r in level_data['per_equation'] if r['difficulty'] == diff]
            rates = [r['recovery_rate'] for r in diff_eqs if r['recovery_rate'] is not None]
            neds = [r['mean_ned'] for r in diff_eqs if r['mean_ned'] is not None]
            curve.append({
                'sigma': sigma,
                'recovery_rate': float(np.mean(rates)) if rates else None,
                'mean_ned': float(np.mean(neds)) if neds else None,
            })
        difficulty_curves[diff] = curve

    report = {
        'description': 'Noise robustness test: multiplicative Gaussian noise y*(1+ε)',
        'noise_levels': NOISE_LEVELS,
        'n_repeats': N_REPEATS,
        'n_per_difficulty': n_per_difficulty,
        'seed': seed,
        'overall_results': {k: {kk: v[kk] for kk in ['sigma', 'n_equations', 'overall_recovery_rate', 'overall_mean_ned', 'overall_mean_r2_log', 'time_s']} for k, v in all_results.items()},
        'difficulty_curves': difficulty_curves,
        'detailed_results': all_results,
    }

    return report


def plot_noise_robustness(report: dict, output_path: str):
    """Plot noise degradation curves.

    Parameters
    ----------
    report : dict
        Output from run_noise_robustness_test.
    output_path : str
        Path to save PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sigmas = report['noise_levels']

    # Overall curve
    rates = [report['overall_results'][str(s)]['overall_recovery_rate'] for s in sigmas]
    neds = [report['overall_results'][str(s)]['overall_mean_ned'] for s in sigmas]

    ax1.plot(sigmas, [r * 100 if r is not None else 0 for r in rates],
             'ko-', linewidth=2, markersize=8, label='Overall')

    # Per-difficulty curves
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    for diff, curve in report['difficulty_curves'].items():
        d_rates = [p['recovery_rate'] for p in curve]
        ax1.plot(sigmas, [r * 100 if r is not None else 0 for r in d_rates],
                 'o--', color=colors[diff], linewidth=1.5, markersize=6, label=diff.capitalize())

    ax1.set_xlabel('Noise level σ', fontsize=12)
    ax1.set_ylabel('Exact recovery rate (%)', fontsize=12)
    ax1.set_title('Recovery Rate vs. Noise Level', fontsize=13)
    ax1.set_ylim(-5, 105)
    ax1.axvline(x=0.83, color='gray', linestyle=':', label='Zheng 2024 worst case')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # NED curve
    ax2.plot(sigmas, [n if n is not None else 0 for n in neds],
             'ko-', linewidth=2, markersize=8, label='Overall')
    for diff, curve in report['difficulty_curves'].items():
        d_neds = [p['mean_ned'] for p in curve]
        ax2.plot(sigmas, [n if n is not None else 0 for n in d_neds],
                 'o--', color=colors[diff], linewidth=1.5, markersize=6, label=diff.capitalize())

    ax2.set_xlabel('Noise level σ', fontsize=12)
    ax2.set_ylabel('Mean NED', fontsize=12)
    ax2.set_title('Exponent Error vs. Noise Level', fontsize=13)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='Exact recovery threshold')
    ax2.axvline(x=0.83, color='gray', linestyle=':', label='Zheng 2024 worst case')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {output_path}")


def main():
    print("=" * 60)
    print("TEST 3: Noise Robustness")
    print("=" * 60)

    report = run_noise_robustness_test(n_per_difficulty=10, seed=42)

    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'noise_robustness_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out_path}")

    fig_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'noise_robustness_curve.png')
    plot_noise_robustness(report, fig_path)

    return report


if __name__ == '__main__':
    main()
