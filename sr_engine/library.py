"""Monomial library generation for symbolic regression.

Generates candidate monomials x0^a * x1^b * ... with configurable
exponent ranges and complexity limits. Optionally filters by
dimensional consistency using Buckingham Pi analysis.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

from .buckingham_pi import check_dimensional_consistency


def build_monomial_library(
    n_features: int,
    exponent_range: tuple[int, int] = (-3, 3),
    max_active_vars: int = 4,
    max_total_degree: int = 6,
    dim_matrix: Optional[np.ndarray] = None,
    target_dim: Optional[np.ndarray] = None,
    include_zero: bool = True,
) -> np.ndarray:
    """Generate all valid monomial exponent vectors.

    Parameters
    ----------
    n_features : int
        Number of input variables.
    exponent_range : tuple of int
        (min_exp, max_exp) inclusive range for each exponent.
    max_active_vars : int
        Maximum number of variables with nonzero exponent in a monomial.
    max_total_degree : int
        Maximum sum of absolute exponents.
    dim_matrix : np.ndarray, optional, shape (n_dims, n_features)
        If provided along with target_dim, only dimensionally valid
        monomials are kept.
    target_dim : np.ndarray, optional, shape (n_dims,)
        Dimensional vector of the target variable.
    include_zero : bool
        Whether to include the trivial all-zeros monomial (constant term).

    Returns
    -------
    np.ndarray, shape (n_monomials, n_features)
        Each row is an exponent vector for one monomial.
    """
    lo, hi = exponent_range
    exp_values = list(range(lo, hi + 1))

    monomials = []

    for exps in product(exp_values, repeat=n_features):
        exps_arr = np.array(exps, dtype=float)

        # Skip all-zeros if not wanted
        if not include_zero and np.all(exps_arr == 0):
            continue

        # Complexity filters
        n_active = np.count_nonzero(exps_arr)
        if n_active > max_active_vars:
            continue

        total_degree = np.sum(np.abs(exps_arr))
        if total_degree > max_total_degree:
            continue

        # Dimensional filter
        if dim_matrix is not None and target_dim is not None:
            if not check_dimensional_consistency(exps_arr, dim_matrix, target_dim):
                continue

        monomials.append(exps_arr)

    if not monomials:
        return np.empty((0, n_features))

    return np.array(monomials)


def evaluate_monomials(
    X: np.ndarray,
    exponent_matrix: np.ndarray,
) -> np.ndarray:
    """Evaluate monomial library on data points.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data (must be positive for log-space computation).
    exponent_matrix : np.ndarray, shape (n_monomials, n_features)
        Each row defines a monomial via its exponents.

    Returns
    -------
    np.ndarray, shape (n_samples, n_monomials)
        Evaluated monomial values. In log-space this becomes
        log(monomial_j) = sum_i exponent_ji * log(X_i).
    """
    log_X = np.log(np.abs(X) + 1e-300)  # Avoid log(0)
    # (n_samples, n_features) @ (n_features, n_monomials) -> (n_samples, n_monomials)
    return log_X @ exponent_matrix.T


def build_logspace_design_matrix(
    X: np.ndarray,
    include_intercept: bool = True,
) -> np.ndarray:
    """Build simple log-space design matrix for power-law regression.

    For the standard case where each feature appears once with unknown
    exponent: log(y) = log(C) + a0*log(x0) + a1*log(x1) + ...

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data (must be positive).
    include_intercept : bool
        Whether to prepend a column of ones.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features + 1) or (n_samples, n_features)
        Log-transformed design matrix.
    """
    log_X = np.log(np.abs(X) + 1e-300)
    if include_intercept:
        return np.column_stack([np.ones(len(log_X)), log_X])
    return log_X
