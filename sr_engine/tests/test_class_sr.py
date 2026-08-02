"""Tests for Class-SR hierarchical symbolic regression."""

import numpy as np
import pytest

from sr_engine.class_sr import (
    class_sr_fit,
    select_best_model,
    _neg_log_likelihood,
    _profile_geometry_params,
    result_to_dict,
    ClassSRResult,
)


def _make_synthetic(
    n_geom: int = 10,
    n_per_geom: int = 50,
    true_exponents: np.ndarray = None,
    noise_std: float = 0.05,
    seed: int = 42,
):
    """Generate synthetic Class-SR data.

    Model: log(y_ij) = log(C_i) + a*log(X1_ij) + b*log(X2_ij) + eps
    """
    rng = np.random.default_rng(seed)

    if true_exponents is None:
        true_exponents = np.array([-0.25, -0.5])

    p = len(true_exponents)
    n = n_geom * n_per_geom
    log_C_true = rng.uniform(-2, 2, size=n_geom)

    log_X = np.zeros((n, p))
    log_y = np.zeros(n)
    group_ids = np.zeros(n, dtype=int)

    for i in range(n_geom):
        sl = slice(i * n_per_geom, (i + 1) * n_per_geom)
        group_ids[sl] = i
        log_X[sl] = rng.uniform(2, 12, size=(n_per_geom, p))
        log_y[sl] = (
            log_C_true[i]
            + log_X[sl] @ true_exponents
            + rng.normal(0, noise_std, size=n_per_geom)
        )

    return log_X, log_y, group_ids, log_C_true, true_exponents


class TestClassSRSynthetic:
    """Synthetic data tests — Class-SR must recover known exponents."""

    def test_recovers_exponents_within_5pct(self):
        """10 geometries, known exponents → recovered within 5% error."""
        true_exp = np.array([-0.25, -0.50])
        log_X, log_y, gids, _, _ = _make_synthetic(
            n_geom=10, n_per_geom=50, true_exponents=true_exp, noise_std=0.03
        )
        res = class_sr_fit(log_y, log_X, gids, ["Re", "Pi_gap"], n_bootstrap=0)

        recovered = np.array(list(res.global_exponents.values()))
        rel_error = np.abs(recovered - true_exp) / np.abs(true_exp)
        assert np.all(rel_error < 0.05), (
            f"Exponents {recovered} too far from {true_exp}, rel error {rel_error}"
        )

    def test_r2_logspace_high(self):
        """Synthetic low-noise data → R² logspace > 0.95."""
        log_X, log_y, gids, _, _ = _make_synthetic(noise_std=0.02)
        res = class_sr_fit(log_y, log_X, gids, ["a", "b"], n_bootstrap=0)
        assert res.r2_logspace > 0.95, f"R² logspace = {res.r2_logspace}"

    def test_prefactors_recovered(self):
        """Per-geometry prefactors should be close to true values."""
        log_X, log_y, gids, log_C_true, _ = _make_synthetic(
            n_geom=5, n_per_geom=100, noise_std=0.02
        )
        res = class_sr_fit(log_y, log_X, gids, ["a", "b"], n_bootstrap=0)

        for i in range(5):
            recovered_log_C = res.prefactors[str(i)]["log_C"]
            assert abs(recovered_log_C - log_C_true[i]) < 0.15, (
                f"Geom {i}: log_C recovered={recovered_log_C:.3f}, "
                f"true={log_C_true[i]:.3f}"
            )


class TestClassSRDegenerate:
    """Edge case: single geometry → should collapse to pooled OLS."""

    def test_single_geometry(self):
        """1 geometry → same as ordinary log-space OLS."""
        rng = np.random.default_rng(123)
        n = 80
        true_exp = np.array([-0.3])
        log_X = rng.uniform(3, 10, size=(n, 1))
        log_y = 1.5 + log_X @ true_exp + rng.normal(0, 0.02, n)
        gids = np.zeros(n, dtype=int)

        res = class_sr_fit(log_y, log_X, gids, ["Re"], n_bootstrap=0)

        assert res.n_geometries == 1
        recovered = list(res.global_exponents.values())[0]
        assert abs(recovered - (-0.3)) < 0.02, f"Recovered {recovered}"

    def test_single_geometry_r2(self):
        """Single geometry R² should match simple OLS."""
        rng = np.random.default_rng(77)
        n = 100
        log_X = rng.uniform(2, 8, size=(n, 2))
        true_exp = np.array([-0.4, 0.6])
        log_y = 0.5 + log_X @ true_exp + rng.normal(0, 0.05, n)
        gids = np.zeros(n, dtype=int)

        res = class_sr_fit(log_y, log_X, gids, ["a", "b"], n_bootstrap=0)
        assert res.r2_logspace > 0.95


class TestBootstrapCI:
    """Bootstrap CI tests."""

    def test_ci_bootstrap_contains_true(self):
        """Bootstrap 95% CI should contain true exponent value."""
        true_exp = np.array([-0.25])
        rng = np.random.default_rng(42)
        n_geom, n_per = 4, 20
        n = n_geom * n_per
        gids = np.repeat(np.arange(n_geom), n_per)
        log_X = rng.uniform(3, 10, size=(n, 1))
        log_C = rng.uniform(-1, 1, size=n_geom)
        log_y = np.zeros(n)
        for i in range(n_geom):
            mask = gids == i
            log_y[mask] = log_C[i] + log_X[mask] @ true_exp + rng.normal(0, 0.15, n_per)

        res = class_sr_fit(log_y, log_X, gids, ["Re"], n_bootstrap=500, seed=42)

        ci = res.ci_bootstrap["Re"]
        assert ci[0] < ci[1], f"CI collapsed: {ci}"
        assert ci[0] < true_exp[0] < ci[1], (
            f"CI {ci} does not contain true {true_exp[0]}"
        )

    def test_ci_does_not_contain_zero_for_strong_signal(self):
        """For a clear signal, CI should not span zero."""
        true_exp = np.array([-0.5])
        rng = np.random.default_rng(99)
        n = 300
        gids = np.repeat(np.arange(6), 50)
        log_X = rng.uniform(3, 12, size=(n, 1))
        log_C = rng.uniform(-1, 1, size=6)
        log_y = np.zeros(n)
        for i in range(6):
            mask = gids == i
            log_y[mask] = log_C[i] + log_X[mask] @ true_exp + rng.normal(0, 0.02, 50)

        res = class_sr_fit(log_y, log_X, gids, ["Re"], n_bootstrap=500, seed=99)
        ci = res.ci_bootstrap["Re"]
        assert ci[1] < 0, f"CI upper bound {ci[1]} should be < 0"


class TestModelSelection:
    """AIC/BIC model selection tests."""

    def test_selects_correct_features(self):
        """Given 3 candidates where only 2 are active, BIC selects the 2."""
        rng = np.random.default_rng(55)
        n = 300
        gids = np.repeat(np.arange(6), 50)
        true_exp = np.array([-0.3, 0.5])
        log_X_active = rng.uniform(3, 10, size=(n, 2))
        log_X_noise = rng.uniform(3, 10, size=(n, 1))  # irrelevant
        log_X_full = np.column_stack([log_X_active, log_X_noise])
        log_C = rng.uniform(-1, 1, size=6)
        log_y = np.zeros(n)
        for i in range(6):
            mask = gids == i
            log_y[mask] = (
                log_C[i] + log_X_active[mask] @ true_exp + rng.normal(0, 0.03, 50)
            )

        _, selected = select_best_model(
            log_y, log_X_full, gids,
            ["Re", "Pi_gap", "noise"],
            max_features=3, criterion="bic", seed=55,
        )
        # The noise feature should NOT be selected
        assert "noise" not in selected, f"Selected {selected} includes noise"
        assert "Re" in selected and "Pi_gap" in selected


class TestSerialization:
    """Test JSON serialization."""

    def test_result_to_dict_roundtrip(self):
        """result_to_dict produces JSON-serializable output."""
        import json
        log_X, log_y, gids, _, _ = _make_synthetic(n_geom=3, n_per_geom=30)
        res = class_sr_fit(log_y, log_X, gids, ["a", "b"], n_bootstrap=100)
        d = result_to_dict(res)
        s = json.dumps(d)
        assert len(s) > 50
        loaded = json.loads(s)
        assert "global_exponents" in loaded


class TestLOSOCV:
    """LOSO cross-validation tests."""

    def test_loso_r2_reasonable(self):
        """LOSO-CV R² should be > -1 for clean synthetic data."""
        log_X, log_y, gids, _, _ = _make_synthetic(
            n_geom=8, n_per_geom=40, noise_std=0.05
        )
        res = class_sr_fit(log_y, log_X, gids, ["a", "b"], n_bootstrap=0)
        assert res.r2_loso_cv > -1.0, f"LOSO R² = {res.r2_loso_cv}"
