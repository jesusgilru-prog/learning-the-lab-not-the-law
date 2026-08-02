"""Sparse regression fitting for power-law symbolic regression.

Fits log(y) = log(C) + a*log(x0) + b*log(x1) + ... using multiple
regression methods with cross-validation and bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, HuberRegressor, RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class FitResult:
    """Result of a sparse regression fit.

    Attributes
    ----------
    method : str
        Name of the regression method used.
    intercept : float
        Intercept in log-space (log(C)).
    coefficient : float
        Multiplicative constant C = exp(intercept).
    exponents : np.ndarray
        Fitted exponents for each feature.
    r2_logspace : float
        R² in log-space.
    r2_original : float
        R² in original space.
    ci_low : np.ndarray
        Lower 95% confidence interval for exponents (bootstrap).
    ci_high : np.ndarray
        Upper 95% confidence interval for exponents (bootstrap).
    residuals : np.ndarray
        Residuals in log-space.
    """
    method: str
    intercept: float
    coefficient: float
    exponents: np.ndarray
    r2_logspace: float
    r2_original: float
    ci_low: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_high: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score, handling edge cases."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _bootstrap_ci(
    log_X: np.ndarray,
    log_y: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for exponents.

    Parameters
    ----------
    log_X : np.ndarray, shape (n_samples, n_features)
        Log-transformed features (no intercept column).
    log_y : np.ndarray, shape (n_samples,)
        Log-transformed target.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 for 95% CI).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ci_low, ci_high : np.ndarray
        Lower and upper CI bounds for each exponent.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(log_y)
    n_features = log_X.shape[1]
    boot_coefs = np.zeros((n_bootstrap, n_features))

    A = np.column_stack([np.ones(n_samples), log_X])

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        Ab = A[idx]
        yb = log_y[idx]
        try:
            coeffs = np.linalg.lstsq(Ab, yb, rcond=None)[0]
            boot_coefs[b] = coeffs[1:]  # skip intercept
        except np.linalg.LinAlgError:
            boot_coefs[b] = np.nan

    # Remove failed fits
    valid = ~np.any(np.isnan(boot_coefs), axis=1)
    boot_coefs = boot_coefs[valid]

    if len(boot_coefs) < 10:
        return np.full(n_features, np.nan), np.full(n_features, np.nan)

    ci_low = np.percentile(boot_coefs, 100 * alpha / 2, axis=0)
    ci_high = np.percentile(boot_coefs, 100 * (1 - alpha / 2), axis=0)

    return ci_low, ci_high


def sparse_fit_logspace(
    X: np.ndarray,
    y: np.ndarray,
    methods: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    cv_folds: int = 5,
    seed: int = 42,
) -> list[FitResult]:
    """Fit power-law model in log-space using multiple sparse methods.

    Transforms to log(y) = log(C) + Σ a_i * log(x_i) and fits with
    multiple regression methods. Returns results sorted by R² (best first).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features (must be positive).
    y : np.ndarray, shape (n_samples,)
        Target values (must be positive).
    methods : list of str, optional
        Regression methods to use. Default: all available.
        Options: 'ols', 'lasso', 'ridge', 'huber', 'ransac'.
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals.
    cv_folds : int
        Number of cross-validation folds for Lasso/Ridge.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of FitResult
        Results for each method, sorted by R² (descending).
    """
    if methods is None:
        methods = ["ols", "lasso", "ridge", "huber", "ransac"]

    rng = np.random.default_rng(seed)

    # Prepare log-space data, handling zeros/negatives
    mask = np.all(X > 0, axis=1) & (y > 0)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(y_clean) < 10:
        # Try absolute values
        mask = np.all(np.abs(X) > 1e-300, axis=1) & (np.abs(y) > 1e-300)
        X_clean = np.abs(X[mask])
        y_clean = np.abs(y[mask])

    if len(y_clean) < 10:
        return []

    log_X = np.log(X_clean)
    log_y = np.log(y_clean)

    n_features = X_clean.shape[1]
    results = []

    for method in methods:
        try:
            result = _fit_single_method(
                log_X, log_y, X_clean, y_clean, method, cv_folds, seed
            )
            results.append(result)
        except Exception:
            continue

    # Compute bootstrap CI once (method-independent, uses OLS)
    ci_low, ci_high = _bootstrap_ci(log_X, log_y, n_bootstrap, rng=rng)
    for r in results:
        r.ci_low = ci_low
        r.ci_high = ci_high

    # Sort by R² in log-space (best first)
    results.sort(key=lambda r: r.r2_logspace, reverse=True)

    return results


def _fit_single_method(
    log_X: np.ndarray,
    log_y: np.ndarray,
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    method: str,
    cv_folds: int,
    seed: int,
) -> FitResult:
    """Fit a single regression method in log-space.

    Parameters
    ----------
    log_X : np.ndarray, shape (n_samples, n_features)
        Log-transformed features.
    log_y : np.ndarray, shape (n_samples,)
        Log-transformed target.
    X_orig : np.ndarray
        Original-space features (for R² computation).
    y_orig : np.ndarray
        Original-space target.
    method : str
        One of 'ols', 'lasso', 'ridge', 'huber', 'ransac'.
    cv_folds : int
        CV folds for regularized methods.
    seed : int
        Random seed.

    Returns
    -------
    FitResult
    """
    n_samples = len(log_y)
    cv = min(cv_folds, n_samples)

    if method == "ols":
        A = np.column_stack([np.ones(n_samples), log_X])
        coeffs = np.linalg.lstsq(A, log_y, rcond=None)[0]
        intercept = coeffs[0]
        exponents = coeffs[1:]

    elif method == "lasso":
        model = LassoCV(
            cv=cv, max_iter=10000, random_state=seed,
            alphas=np.logspace(-8, 1, 50),
        )
        model.fit(log_X, log_y)
        intercept = model.intercept_
        exponents = model.coef_

    elif method == "ridge":
        model = RidgeCV(
            cv=cv,
            alphas=np.logspace(-8, 4, 50),
        )
        model.fit(log_X, log_y)
        intercept = model.intercept_
        exponents = model.coef_

    elif method == "huber":
        model = HuberRegressor(max_iter=1000, epsilon=1.35)
        model.fit(log_X, log_y)
        intercept = model.intercept_
        exponents = model.coef_

    elif method == "ransac":
        base = LinearRegression()
        model = RANSACRegressor(
            estimator=base, random_state=seed,
            max_trials=500, min_samples=max(0.5, 10 / n_samples),
        )
        model.fit(log_X, log_y)
        intercept = model.estimator_.intercept_
        exponents = model.estimator_.coef_

    else:
        raise ValueError(f"Unknown method: {method}")

    exponents = np.asarray(exponents, dtype=float)
    intercept = float(intercept)

    # R² in log-space
    log_y_pred = intercept + log_X @ exponents
    r2_log = _compute_r2(log_y, log_y_pred)

    # R² in original space
    y_pred_orig = np.exp(intercept) * np.prod(X_orig ** exponents, axis=1)
    r2_orig = _compute_r2(y_orig, y_pred_orig)

    residuals = log_y - log_y_pred

    return FitResult(
        method=method,
        intercept=intercept,
        coefficient=np.exp(intercept),
        exponents=exponents,
        r2_logspace=r2_log,
        r2_original=r2_orig,
        ci_low=np.array([]),
        ci_high=np.array([]),
        residuals=residuals,
    )
