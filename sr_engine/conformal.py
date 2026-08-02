"""Conformal Prediction for Class-SR models.

Three variants:
  A) Split conformal (vanilla) — global nonconformity quantile
  B) Mondrian (class-conditional) — per-geometry quantile calibration
  C) Normalized (locally-adaptive) — scores normalized by per-geometry σ_i
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Result of conformal prediction evaluation."""
    variant: str
    alpha: float
    global_coverage: float
    mean_width_logspace: float
    per_geometry_coverage: dict  # {geom: coverage}
    per_geometry_width: dict  # {geom: width}
    q_hat: float | dict  # scalar for split/normalized, dict for Mondrian
    min_logo_coverage: float
    logo_results: dict


def split_conformal(abs_residuals, geom_ids, alpha=0.10, n_splits=200, seed=42):
    """Vanilla split conformal with repeated stratified splits."""
    n = len(abs_residuals)
    rng = np.random.default_rng(seed)
    unique_geoms = np.unique(geom_ids)

    coverages = []
    widths = []

    for _ in range(n_splits):
        cal_mask = np.zeros(n, dtype=bool)
        for g in unique_geoms:
            g_idx = np.where(geom_ids == g)[0]
            n_cal = max(1, len(g_idx) // 2)
            chosen = rng.choice(g_idx, size=n_cal, replace=False)
            cal_mask[chosen] = True

        test_mask = ~cal_mask
        cal_scores = abs_residuals[cal_mask]
        n_cal = len(cal_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_hat = np.quantile(cal_scores, q_level)

        test_scores = abs_residuals[test_mask]
        coverage = np.mean(test_scores <= q_hat)
        coverages.append(coverage)
        widths.append(2 * q_hat)

    return np.mean(coverages), np.mean(widths), np.array(coverages), np.array(widths)


def mondrian_conformal(abs_residuals, geom_ids, alpha=0.10):
    """Mondrian (class-conditional) conformal: per-geometry calibration.

    For each geometry, uses LOO within that geometry to compute coverage.
    Calibration quantile is geometry-specific.
    """
    unique_geoms = np.unique(geom_ids)
    per_geom_coverage = {}
    per_geom_width = {}
    q_hats = {}

    for g in unique_geoms:
        mask = geom_ids == g
        scores = abs_residuals[mask]
        n = len(scores)

        if n < 3:
            # Too few: use max as conservative bound
            q_hat = float(np.max(scores)) * 1.5
            per_geom_coverage[g] = 1.0
            per_geom_width[g] = 2 * q_hat
            q_hats[g] = q_hat
            continue

        # LOO conformal within this geometry
        covered = 0
        for i in range(n):
            cal = np.delete(scores, i)
            q_level = min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal))
            q_hat_loo = np.quantile(cal, q_level)
            if scores[i] <= q_hat_loo:
                covered += 1

        per_geom_coverage[g] = covered / n

        # Full calibration quantile for this geometry
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q_hat = np.quantile(scores, q_level)
        per_geom_width[g] = 2 * q_hat
        q_hats[g] = q_hat

    global_coverage = np.mean([per_geom_coverage[g] for g in unique_geoms])
    mean_width = np.mean([per_geom_width[g] for g in unique_geoms])

    return global_coverage, mean_width, per_geom_coverage, per_geom_width, q_hats


def normalized_conformal(abs_residuals, geom_ids, sigma_per_geometry, alpha=0.10, n_splits=200, seed=42):
    """Normalized conformal: scores = |residual| / σ_i.

    Intervals expand for noisy geometries and shrink for clean ones.
    """
    n = len(abs_residuals)
    rng = np.random.default_rng(seed)
    unique_geoms = np.unique(geom_ids)

    # Compute normalized scores
    sigmas = np.array([sigma_per_geometry.get(g, 1.0) for g in geom_ids])
    # Floor sigma to avoid division by near-zero
    sigmas = np.maximum(sigmas, 0.01)
    normalized_scores = abs_residuals / sigmas

    coverages = []
    widths_per_geom = {g: [] for g in unique_geoms}

    for _ in range(n_splits):
        cal_mask = np.zeros(n, dtype=bool)
        for g in unique_geoms:
            g_idx = np.where(geom_ids == g)[0]
            n_cal = max(1, len(g_idx) // 2)
            chosen = rng.choice(g_idx, size=n_cal, replace=False)
            cal_mask[chosen] = True

        test_mask = ~cal_mask
        cal_norm_scores = normalized_scores[cal_mask]
        n_cal = len(cal_norm_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_hat_norm = np.quantile(cal_norm_scores, q_level)

        # Coverage: test point covered if normalized_score <= q_hat_norm
        test_norm_scores = normalized_scores[test_mask]
        coverage = np.mean(test_norm_scores <= q_hat_norm)
        coverages.append(coverage)

        # Width per geometry: 2 * q_hat_norm * sigma_i
        for g in unique_geoms:
            sigma_g = sigma_per_geometry.get(g, 1.0)
            widths_per_geom[g].append(2 * q_hat_norm * max(sigma_g, 0.01))

    mean_coverage = np.mean(coverages)
    per_geom_width = {g: np.mean(widths_per_geom[g]) for g in unique_geoms}
    mean_width = np.mean(list(per_geom_width.values()))

    # Per-geometry coverage via LOO on normalized scores
    per_geom_coverage = {}
    for g in unique_geoms:
        mask = geom_ids == g
        g_scores = normalized_scores[mask]
        ng = len(g_scores)
        if ng < 3:
            per_geom_coverage[g] = 1.0
            continue
        # Use full dataset except this geometry as calibration
        cal_scores = normalized_scores[~mask]
        nc = len(cal_scores)
        q_level = min(1.0, np.ceil((nc + 1) * (1 - alpha)) / nc)
        q_hat = np.quantile(cal_scores, q_level)
        per_geom_coverage[g] = float(np.mean(g_scores <= q_hat))

    return mean_coverage, mean_width, per_geom_coverage, per_geom_width, np.mean(coverages)


def logo_conformal(abs_residuals, geom_ids, alpha=0.10, sigma_per_geometry=None, variant="split"):
    """Leave-One-Geometry-Out for any variant.

    variant: "split" (vanilla), "normalized" (uses sigma normalization)
    """
    unique_geoms = np.unique(geom_ids)
    results = {}

    for g in unique_geoms:
        test_mask = geom_ids == g
        cal_mask = ~test_mask

        if variant == "normalized" and sigma_per_geometry is not None:
            sigmas_cal = np.array([max(sigma_per_geometry.get(gid, 1.0), 0.01)
                                   for gid in geom_ids[cal_mask]])
            sigmas_test = np.array([max(sigma_per_geometry.get(gid, 1.0), 0.01)
                                    for gid in geom_ids[test_mask]])
            cal_scores = abs_residuals[cal_mask] / sigmas_cal
            test_scores = abs_residuals[test_mask] / sigmas_test
        else:
            cal_scores = abs_residuals[cal_mask]
            test_scores = abs_residuals[test_mask]

        n_cal = len(cal_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        q_hat = np.quantile(cal_scores, q_level)

        coverage = float(np.mean(test_scores <= q_hat)) if len(test_scores) > 0 else 0.0

        # Width depends on variant
        if variant == "normalized" and sigma_per_geometry is not None:
            sigma_g = max(sigma_per_geometry.get(g, 1.0), 0.01)
            width = 2 * q_hat * sigma_g
        else:
            width = 2 * q_hat

        results[g] = {
            "coverage": coverage,
            "n_test": int(np.sum(test_mask)),
            "q_hat": float(q_hat),
            "width_logspace": float(width),
        }

    return results
