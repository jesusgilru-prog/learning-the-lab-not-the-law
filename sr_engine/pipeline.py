"""End-to-end symbolic regression pipeline.

Orchestrates dimensional analysis, library construction, sparse fitting,
and evaluation into a single `audit_law` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .buckingham_pi import find_pi_groups, filter_dimensionally_valid
from .library import build_monomial_library, build_logspace_design_matrix
from .sparse_fit import sparse_fit_logspace, FitResult
from .metrics import (
    r_squared,
    normalized_exponent_distance,
    complexity_score,
    round_to_nearest_fraction,
)


@dataclass
class AuditResult:
    """Result of a symbolic regression audit.

    Attributes
    ----------
    equation_id : str
        Identifier for the equation.
    equation_str : str
        Human-readable recovered equation string.
    coefficient : float
        Multiplicative constant C.
    exponents : np.ndarray
        Recovered exponents for each feature.
    exponents_rounded : np.ndarray
        Exponents rounded to nearest simple fraction.
    r2_logspace : float
        R² in log-space.
    r2_original : float
        R² in original space.
    ned : float or None
        Normalized Exponent Distance (if ground truth provided).
    complexity : float
        Complexity score of the recovered law.
    ci_low : np.ndarray
        Lower 95% CI for exponents.
    ci_high : np.ndarray
        Upper 95% CI for exponents.
    residuals : np.ndarray
        Residuals in log-space.
    best_method : str
        Name of the best regression method.
    all_fits : list of FitResult
        Results from all methods.
    pi_groups : list of dict or None
        Dimensionless Pi groups (if dimensions provided).
    is_exact_recovery : bool
        Whether NED < 0.1 (exact recovery).
    feature_names : list of str
        Names of the features.
    """
    equation_id: str = ""
    equation_str: str = ""
    coefficient: float = 1.0
    exponents: np.ndarray = field(default_factory=lambda: np.array([]))
    exponents_rounded: np.ndarray = field(default_factory=lambda: np.array([]))
    r2_logspace: float = 0.0
    r2_original: float = 0.0
    ned: Optional[float] = None
    complexity: float = 0.0
    ci_low: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_high: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    best_method: str = ""
    all_fits: list = field(default_factory=list)
    pi_groups: Optional[list] = None
    is_exact_recovery: bool = False
    feature_names: list = field(default_factory=list)


def _format_equation(
    coefficient: float,
    exponents: np.ndarray,
    feature_names: list[str],
) -> str:
    """Format a power-law equation as a human-readable string.

    Parameters
    ----------
    coefficient : float
        Multiplicative constant.
    exponents : np.ndarray
        Exponents per feature.
    feature_names : list of str
        Feature names.

    Returns
    -------
    str
        E.g., "9.81 * x0^1.0 * x1^2.0"
    """
    parts = []

    # Format coefficient
    if abs(coefficient - 1.0) > 0.001:
        if abs(coefficient - round(coefficient)) < 0.001 and abs(coefficient) < 1e6:
            parts.append(f"{int(round(coefficient))}")
        else:
            parts.append(f"{coefficient:.6g}")

    for name, exp in zip(feature_names, exponents):
        if abs(exp) < 0.01:
            continue
        if abs(exp - 1.0) < 0.01:
            parts.append(name)
        elif abs(exp - round(exp)) < 0.01:
            parts.append(f"{name}^{int(round(exp))}")
        else:
            parts.append(f"{name}^{exp:.3f}")

    return " * ".join(parts) if parts else "1"


def audit_law(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
    dimensions: Optional[np.ndarray] = None,
    target_dim: Optional[np.ndarray] = None,
    ground_truth: Optional[dict] = None,
    equation_id: str = "",
    methods: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> AuditResult:
    """Run full symbolic regression audit on a dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features.
    y : np.ndarray, shape (n_samples,)
        Target values.
    feature_names : list of str, optional
        Names for each feature. Defaults to x0, x1, ...
    dimensions : np.ndarray, optional, shape (n_dims, n_features)
        Dimensional matrix of input variables.
    target_dim : np.ndarray, optional, shape (n_dims,)
        Dimensional vector of the target variable.
    ground_truth : dict, optional
        If provided, must have keys 'exponents' (list/array) and optionally
        'coefficient' (float). Used to compute NED.
    equation_id : str
        Identifier for this equation.
    methods : list of str, optional
        Regression methods. Default: all.
    n_bootstrap : int
        Bootstrap resamples for CI.
    seed : int
        Random seed.

    Returns
    -------
    AuditResult
        Complete audit results including equation, metrics, and diagnostics.
    """
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # Step 1: Dimensional analysis (if dimensions provided)
    pi_groups = None
    if dimensions is not None and target_dim is not None:
        pi_groups_raw = find_pi_groups(
            dimensions,
            variable_names=feature_names,
        )
        pi_groups = pi_groups_raw

    # Step 2: Sparse fitting in log-space
    fits = sparse_fit_logspace(
        X, y,
        methods=methods,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if not fits:
        return AuditResult(
            equation_id=equation_id,
            equation_str="FAILED: no valid fit",
            feature_names=feature_names,
        )

    # Select best fit
    best = fits[0]  # Already sorted by R²

    # Step 3: Round exponents to nearest simple fractions
    exponents_rounded = np.array([
        round_to_nearest_fraction(e) for e in best.exponents
    ])

    # Step 4: Compute metrics
    ned = None
    is_exact = False
    if ground_truth is not None and "exponents" in ground_truth:
        gt_exp = np.array(ground_truth["exponents"], dtype=float)
        ned = normalized_exponent_distance(exponents_rounded, gt_exp)
        is_exact = ned < 0.1

    compl = complexity_score(exponents_rounded, best.coefficient)
    eq_str = _format_equation(best.coefficient, exponents_rounded, feature_names)

    return AuditResult(
        equation_id=equation_id,
        equation_str=eq_str,
        coefficient=best.coefficient,
        exponents=best.exponents,
        exponents_rounded=exponents_rounded,
        r2_logspace=best.r2_logspace,
        r2_original=best.r2_original,
        ned=ned,
        complexity=compl,
        ci_low=best.ci_low,
        ci_high=best.ci_high,
        residuals=best.residuals,
        best_method=best.method,
        all_fits=fits,
        pi_groups=pi_groups,
        is_exact_recovery=is_exact,
        feature_names=feature_names,
    )
