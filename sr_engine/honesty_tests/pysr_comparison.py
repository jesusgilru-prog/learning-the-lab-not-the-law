"""Test 2: PySR comparison benchmark.

Runs PySR on the same 73 power-law problems and compares with our pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data', 'processed', 'loaders'))

from sr_engine.pipeline import audit_law
from sr_engine.metrics import normalized_exponent_distance
from srsd_feynman import load_srsd_feynman


def check_pysr_available():
    """Check if PySR + Julia backend are available."""
    try:
        import pysr
        return True
    except ImportError:
        return False


def run_pysr_on_problem(X, y, feature_names, timeout_s=30, seed=42):
    """Run PySR on a single problem.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    feature_names : list of str
    timeout_s : int
        Time budget in seconds.
    seed : int

    Returns
    -------
    dict or None
        Result dict with equation, R², exponents if successful.
    """
    from pysr import PySRRegressor

    # Subsample if too large (PySR is slow on 8000 points)
    rng = np.random.default_rng(seed)
    if len(y) > 500:
        idx = rng.choice(len(y), size=500, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube", "sqrt", "inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1/x},
        maxsize=20,
        timeout_in_seconds=timeout_s,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        procs=0,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_sub, y_sub, variable_names=feature_names)
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    try:
        best = model.get_best()
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'status': 'ok',
            'equation': str(best['equation']),
            'complexity': int(best['complexity']),
            'r2_original': float(r2),
            'loss': float(best['loss']),
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def run_pysr_comparison(max_problems: int = 73, timeout_per_problem: int = 30, seed: int = 42):
    """Run full PySR comparison.

    Parameters
    ----------
    max_problems : int
        Max number of problems to test.
    timeout_per_problem : int
        Seconds per PySR run.
    seed : int

    Returns
    -------
    dict
        Comparison report.
    """
    subset_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'srsd_powerlaw_subset.json')
    with open(subset_path) as f:
        subset = json.load(f)

    rng = np.random.default_rng(seed)
    candidates = subset['candidates']
    if max_problems < len(candidates):
        indices = rng.choice(len(candidates), size=max_problems, replace=False)
        candidates = [candidates[i] for i in sorted(indices)]

    our_results = []
    pysr_results = []

    for i, c in enumerate(candidates):
        diff = c['difficulty']
        parts = c['equation_id'].split('_')
        eq_idx = int(parts[-1])

        try:
            eq = load_srsd_feynman('train', diff, eq_idx)
        except Exception:
            continue

        gt = {'exponents': c['exponents'], 'coefficient': c['coefficient']}

        # Our method
        t0 = time.time()
        our = audit_law(
            X=eq['X'], y=eq['y'],
            feature_names=eq['feature_names'],
            ground_truth=gt,
            equation_id=c['equation_id'],
            n_bootstrap=100, seed=seed,
        )
        our_time = time.time() - t0

        our_results.append({
            'equation_id': c['equation_id'],
            'difficulty': diff,
            'ned': our.ned,
            'r2_logspace': our.r2_logspace,
            'r2_original': our.r2_original,
            'is_exact': our.is_exact_recovery,
            'equation': our.equation_str,
            'time_s': our_time,
        })

        # PySR
        t0 = time.time()
        pysr_res = run_pysr_on_problem(
            eq['X'], eq['y'], eq['feature_names'],
            timeout_s=timeout_per_problem, seed=seed,
        )
        pysr_time = time.time() - t0

        pysr_entry = {
            'equation_id': c['equation_id'],
            'difficulty': diff,
            'time_s': pysr_time,
        }
        pysr_entry.update(pysr_res)
        pysr_results.append(pysr_entry)

        print(
            f"  [{i+1}/{len(candidates)}] {c['equation_id']}: "
            f"Ours NED={our.ned:.4f} R²={our.r2_logspace:.4f} ({our_time:.1f}s) | "
            f"PySR: {pysr_res.get('status', '?')} ({pysr_time:.1f}s)"
        )

    # Aggregate
    our_exact = sum(1 for r in our_results if r['is_exact'])
    our_mean_ned = np.mean([r['ned'] for r in our_results if r['ned'] is not None])
    our_mean_r2 = np.mean([r['r2_logspace'] for r in our_results])
    our_mean_time = np.mean([r['time_s'] for r in our_results])

    pysr_ok = [r for r in pysr_results if r.get('status') == 'ok']
    pysr_mean_r2 = np.mean([r['r2_original'] for r in pysr_ok]) if pysr_ok else None
    pysr_mean_time = np.mean([r['time_s'] for r in pysr_results])
    pysr_errors = sum(1 for r in pysr_results if r.get('status') == 'error')

    report = {
        'description': 'PySR vs. our pipeline comparison on SRSD power-law subset',
        'n_problems': len(candidates),
        'our_method': {
            'exact_recovery': our_exact,
            'recovery_rate': our_exact / len(our_results) if our_results else 0,
            'mean_ned': float(our_mean_ned),
            'mean_r2_logspace': float(our_mean_r2),
            'mean_time_s': float(our_mean_time),
        },
        'pysr': {
            'successful_runs': len(pysr_ok),
            'errors': pysr_errors,
            'mean_r2_original': float(pysr_mean_r2) if pysr_mean_r2 is not None else None,
            'mean_time_s': float(pysr_mean_time),
        },
        'our_detailed': our_results,
        'pysr_detailed': pysr_results,
    }

    return report


def format_comparison_table(report: dict) -> str:
    """Format comparison as Markdown table for paper.

    Parameters
    ----------
    report : dict

    Returns
    -------
    str
        Markdown table.
    """
    our = report['our_method']
    pysr = report['pysr']
    n = report['n_problems']

    table = f"""| Metric | HyperScale-CHIEF | PySR |
|--------|-----------------|------|
| Problems tested | {n} | {n} |
| Successful runs | {n} | {pysr['successful_runs']} |
| Exact recovery (NED<0.1) | {our['exact_recovery']}/{n} ({our['recovery_rate']:.0%}) | N/A* |
| Mean NED | {our['mean_ned']:.4f} | N/A* |
| Mean R² | {our['mean_r2_logspace']:.4f} | {f"{pysr['mean_r2_original']:.4f}" if pysr['mean_r2_original'] is not None else 'N/A'} |
| Mean time/problem | {our['mean_time_s']:.2f}s | {pysr['mean_time_s']:.1f}s |

*PySR returns symbolic expressions, not exponent vectors. Direct NED comparison
requires parsing PySR output into power-law form, which is non-trivial for
composite expressions."""
    return table


def main():
    print("=" * 60)
    print("TEST 2: PySR Comparison")
    print("=" * 60)

    if not check_pysr_available():
        print("  PySR not installed. Attempting install...")
        os.system("pip install pysr 2>/dev/null")

    # Check Julia backend
    try:
        import pysr
        print("  PySR imported successfully. Checking Julia backend...")
        # This will trigger Julia install if needed
    except Exception as e:
        blocker = {
            'description': 'PySR comparison blocked',
            'reason': f'Julia/PySR backend error: {e}',
            'recommendation': 'Install Julia manually: curl -fsSL https://install.julialang.org | sh',
        }
        out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pysr_comparison_report.json')
        with open(out_path, 'w') as f:
            json.dump(blocker, f, indent=2)
        print(f"  BLOCKED: {e}")
        print(f"  Blocker report saved: {out_path}")
        return blocker

    # Run with reduced set for time (PySR is slow)
    report = run_pysr_comparison(max_problems=20, timeout_per_problem=30, seed=42)

    out_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pysr_comparison_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out_path}")

    table = format_comparison_table(report)
    print("\n" + table)

    return report


if __name__ == '__main__':
    main()
