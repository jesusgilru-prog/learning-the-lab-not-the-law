"""Metrics for symbolic regression evaluation.

Includes R², Normalized Exponent Distance (NED), and complexity scoring.
"""

from __future__ import annotations

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R².

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R² score. 1.0 is perfect, can be negative for bad fits.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return float(1.0 - ss_res / ss_tot)


def normalized_exponent_distance(
    recovered: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Normalized Exponent Distance (NED) between recovered and true exponents.

    Measures how close the recovered power-law exponents are to the ground
    truth, normalized by the magnitude of the ground truth exponents.

    NED = ||recovered - ground_truth||₂ / max(||ground_truth||₂, 1)

    Parameters
    ----------
    recovered : np.ndarray
        Recovered exponents from SR.
    ground_truth : np.ndarray
        True exponents.

    Returns
    -------
    float
        NED score. 0.0 is perfect recovery. < 0.1 considered exact.
    """
    recovered = np.asarray(recovered, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=float)

    # Pad shorter array with zeros if needed
    max_len = max(len(recovered), len(ground_truth))
    rec = np.zeros(max_len)
    gt = np.zeros(max_len)
    rec[:len(recovered)] = recovered
    gt[:len(ground_truth)] = ground_truth

    dist = np.linalg.norm(rec - gt)
    norm = max(np.linalg.norm(gt), 1.0)

    return float(dist / norm)


def complexity_score(exponents: np.ndarray, coefficient: float = 1.0) -> float:
    """Compute complexity score for a power-law expression.

    Complexity = number of active terms + sum of |exponents| + log_complexity(C).

    Parameters
    ----------
    exponents : np.ndarray
        Exponents in the power-law.
    coefficient : float
        Multiplicative constant C.

    Returns
    -------
    float
        Complexity score. Lower is simpler.
    """
    exponents = np.asarray(exponents, dtype=float)

    n_active = int(np.count_nonzero(np.abs(exponents) > 0.01))
    degree_sum = float(np.sum(np.abs(exponents[np.abs(exponents) > 0.01])))

    # Penalize complex coefficients (far from simple fractions)
    coeff_complexity = 0.0
    if abs(coefficient) > 1e-30:
        log_c = np.log10(abs(coefficient))
        # Simple constants like 1, 2, π, etc. have low complexity
        coeff_complexity = abs(log_c) * 0.1

    return float(n_active + degree_sum + coeff_complexity)


def round_to_nearest_fraction(
    value: float,
    denominators: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    tol: float = 0.08,
) -> float:
    """Round a value to the nearest simple fraction if close enough.

    Parameters
    ----------
    value : float
        Value to round.
    denominators : tuple of int
        Denominators to try (e.g., 2 allows halves, 3 allows thirds).
    tol : float
        Maximum distance to snap to a fraction.

    Returns
    -------
    float
        Rounded value, or original if no close fraction found.
    """
    best = value
    best_dist = tol

    for d in denominators:
        # Try numerators in a reasonable range
        for n in range(-20, 21):
            frac = n / d
            dist = abs(value - frac)
            if dist < best_dist:
                best = frac
                best_dist = dist

    return best
