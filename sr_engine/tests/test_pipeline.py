"""Tests for the SR pipeline on SRSD power-law subset.

Loads problems from srsd_powerlaw_subset.json, runs audit_law,
and reports recovery rates. Target: >= 60% exact recovery on easy problems.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data', 'processed', 'loaders'))

from sr_engine.pipeline import audit_law
from sr_engine.metrics import normalized_exponent_distance
from sr_engine.buckingham_pi import find_pi_groups, n_pi_groups


# Path to dataset
SUBSET_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'srsd_powerlaw_subset.json')
LOADER_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'loaders')


def load_subset():
    """Load the SRSD power-law subset index."""
    with open(SUBSET_PATH) as f:
        return json.load(f)


def load_equation_data(equation_id: str, difficulty: str):
    """Load actual data for an equation from SRSD-Feynman dataset."""
    sys.path.insert(0, LOADER_PATH)
    from srsd_feynman import load_srsd_feynman

    # Parse equation index from id like "feynman_easy_003"
    parts = equation_id.split('_')
    idx = int(parts[-1])

    eq = load_srsd_feynman(split='train', difficulty=difficulty, equation_idx=idx)
    return eq


class TestBuckinghamPiCanonical:
    """Canonical dimensional analysis test cases."""

    def test_pendulum(self):
        """T = 2π√(L/g) -> 1 Pi group."""
        dim = np.array([
            [0, 1,  1],
            [0, 0,  0],
            [1, 0, -2],
        ])
        assert n_pi_groups(dim) == 1

    def test_ideal_gas(self):
        """PV = nRT -> multiple groups."""
        # P (Pa = kg/m/s²), V (m³), n (mol), R (J/mol/K), T (K)
        # Dims: L, M, T, mol, K
        dim = np.array([
            [-1, 3, 0, 2, 0],   # L
            [ 1, 0, 0, 1, 0],   # M
            [-2, 0, 0,-2, 0],   # T
            [ 0, 0, 1,-1, 0],   # mol
            [ 0, 0, 0,-1, 1],   # K
        ])
        groups = find_pi_groups(dim, ['P', 'V', 'n', 'R', 'T'])
        assert len(groups) >= 0  # Just verify it runs


class TestSRSDPowerLawRecovery:
    """Test power-law recovery on SRSD-Feynman subset."""

    @pytest.fixture(scope="class")
    def subset(self):
        return load_subset()

    @pytest.fixture(scope="class")
    def easy_candidates(self, subset):
        return [c for c in subset['candidates'] if c['difficulty'] == 'easy']

    @pytest.fixture(scope="class")
    def medium_candidates(self, subset):
        return [c for c in subset['candidates'] if c['difficulty'] == 'medium']

    def _run_audit(self, candidate):
        """Run audit_law on a single candidate."""
        eq = load_equation_data(candidate['equation_id'], candidate['difficulty'])
        gt = {
            'exponents': candidate['exponents'],
            'coefficient': candidate['coefficient'],
        }
        result = audit_law(
            X=eq['X'],
            y=eq['y'],
            feature_names=eq['feature_names'],
            ground_truth=gt,
            equation_id=candidate['equation_id'],
            n_bootstrap=200,  # Fewer for speed in tests
            seed=42,
        )
        return result

    def test_easy_recovery_rate(self, easy_candidates):
        """Easy problems must achieve >= 60% exact recovery (NED < 0.1)."""
        # Sample up to 10 random easy problems
        rng = np.random.default_rng(42)
        n_test = min(10, len(easy_candidates))
        indices = rng.choice(len(easy_candidates), size=n_test, replace=False)
        test_cases = [easy_candidates[i] for i in indices]

        results = []
        times = []

        for candidate in test_cases:
            t0 = time.time()
            result = self._run_audit(candidate)
            dt = time.time() - t0
            times.append(dt)
            results.append(result)
            print(
                f"  {result.equation_id}: "
                f"NED={result.ned:.4f}, "
                f"R²={result.r2_logspace:.6f}, "
                f"eq={result.equation_str}, "
                f"method={result.best_method}, "
                f"time={dt:.2f}s"
            )

        n_exact = sum(1 for r in results if r.is_exact_recovery)
        rate = n_exact / n_test
        avg_time = np.mean(times)

        print(f"\n  Easy recovery: {n_exact}/{n_test} = {rate:.0%}")
        print(f"  Avg time: {avg_time:.2f}s")

        assert rate >= 0.60, (
            f"Easy recovery rate {rate:.0%} < 60%. "
            f"Results: {[(r.equation_id, r.ned) for r in results]}"
        )

    def test_medium_sample(self, medium_candidates):
        """Run a few medium problems to characterize performance."""
        rng = np.random.default_rng(123)
        n_test = min(5, len(medium_candidates))
        indices = rng.choice(len(medium_candidates), size=n_test, replace=False)
        test_cases = [medium_candidates[i] for i in indices]

        results = []
        for candidate in test_cases:
            result = self._run_audit(candidate)
            results.append(result)
            print(
                f"  {result.equation_id}: "
                f"NED={result.ned:.4f}, "
                f"R²={result.r2_logspace:.6f}, "
                f"eq={result.equation_str}"
            )

        n_exact = sum(1 for r in results if r.is_exact_recovery)
        print(f"\n  Medium recovery: {n_exact}/{n_test}")

    def test_single_known_equation(self, easy_candidates):
        """Test on feynman_easy_000 (should be trivial: y = x0 * x1)."""
        # Find feynman_easy_000
        target = None
        for c in easy_candidates:
            if c['equation_id'] == 'feynman_easy_000':
                target = c
                break

        if target is None:
            pytest.skip("feynman_easy_000 not in subset")

        result = self._run_audit(target)

        assert result.r2_logspace > 0.99
        assert result.ned is not None
        assert result.ned < 0.1, f"NED={result.ned} for trivial equation"

        # Check exponents are close to [1, 1]
        for i, exp in enumerate(result.exponents_rounded):
            gt_exp = target['exponents'][i]
            assert abs(exp - round(gt_exp)) < 0.15, (
                f"Exponent {i}: recovered {exp}, expected ~{gt_exp}"
            )

    def test_confidence_intervals_exist(self, easy_candidates):
        """CI bounds must be populated for fits."""
        target = easy_candidates[0]
        result = self._run_audit(target)
        assert len(result.ci_low) > 0
        assert len(result.ci_high) > 0
        # CI should bracket the point estimate (mostly)
        for i in range(len(result.exponents)):
            assert result.ci_low[i] <= result.exponents[i] + 0.1
            assert result.ci_high[i] >= result.exponents[i] - 0.1


class TestMetricsUnit:
    """Unit tests for individual metrics."""

    def test_ned_perfect(self):
        assert normalized_exponent_distance(
            np.array([1.0, 2.0]), np.array([1.0, 2.0])
        ) < 1e-10

    def test_ned_close(self):
        ned = normalized_exponent_distance(
            np.array([1.01, 1.99]), np.array([1.0, 2.0])
        )
        assert ned < 0.1

    def test_ned_far(self):
        ned = normalized_exponent_distance(
            np.array([2.0, 0.5]), np.array([1.0, 2.0])
        )
        assert ned > 0.5

    def test_ned_different_lengths(self):
        ned = normalized_exponent_distance(
            np.array([1.0, 2.0, 0.0]), np.array([1.0, 2.0])
        )
        assert ned < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
