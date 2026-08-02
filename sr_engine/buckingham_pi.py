"""Buckingham Pi theorem implementation for dimensional analysis.

Given a dimensional matrix (rows = base dimensions like L, M, T;
cols = physical variables), compute the dimensionless Pi groups
using the null space of the dimensional matrix.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sympy import Matrix, Rational


def find_pi_groups(
    dim_matrix: np.ndarray,
    variable_names: Optional[list[str]] = None,
) -> list[dict[str, float]]:
    """Find dimensionless Pi groups via the Buckingham Pi theorem.

    Parameters
    ----------
    dim_matrix : np.ndarray, shape (n_dims, n_vars)
        Dimensional matrix where rows are base dimensions (e.g., [L, M, T])
        and columns are physical variables. Entry (i, j) is the exponent
        of base dimension i in variable j.
    variable_names : list of str, optional
        Names for each variable (column). Defaults to x0, x1, ...

    Returns
    -------
    list of dict
        Each dict maps variable name -> exponent in that Pi group.
        Number of groups = n_vars - rank(dim_matrix).

    Examples
    --------
    Pendulum: T = f(L, g). Variables: [T, L, g] with dims [T, L, M]:
    >>> dim = np.array([
    ...     [0, 1, 1],   # L: T has 0, L has 1, g has 1 (m/s²)
    ...     [0, 0, 0],   # M: all zero
    ...     [1, 0, -2],  # T: T has 1, L has 0, g has -2
    ... ])
    >>> groups = find_pi_groups(dim, ['T', 'L', 'g'])
    """
    n_dims, n_vars = dim_matrix.shape

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_vars)]

    if len(variable_names) != n_vars:
        raise ValueError(
            f"variable_names length ({len(variable_names)}) != "
            f"number of columns ({n_vars})"
        )

    # Use sympy for exact rational null space computation
    M = Matrix(dim_matrix.tolist()).applyfunc(Rational)
    nullspace = M.nullspace()

    pi_groups = []
    for vec in nullspace:
        group = {}
        for j, name in enumerate(variable_names):
            exp = float(vec[j])
            if abs(exp) > 1e-12:
                group[name] = exp
        pi_groups.append(group)

    return pi_groups


def n_pi_groups(dim_matrix: np.ndarray) -> int:
    """Return the number of independent Pi groups.

    Parameters
    ----------
    dim_matrix : np.ndarray, shape (n_dims, n_vars)
        Dimensional matrix.

    Returns
    -------
    int
        n_vars - rank(dim_matrix)
    """
    M = Matrix(dim_matrix.tolist()).applyfunc(Rational)
    return dim_matrix.shape[1] - M.rank()


def check_dimensional_consistency(
    exponents: np.ndarray,
    dim_matrix: np.ndarray,
    target_dim: np.ndarray,
    tol: float = 1e-10,
) -> bool:
    """Check if a monomial x0^a0 * x1^a1 * ... has the target dimensions.

    Parameters
    ----------
    exponents : np.ndarray, shape (n_vars,)
        Exponent for each variable in the monomial.
    dim_matrix : np.ndarray, shape (n_dims, n_vars)
        Dimensional matrix of the input variables.
    target_dim : np.ndarray, shape (n_dims,)
        Dimensional vector of the target variable.
    tol : float
        Tolerance for dimensional matching.

    Returns
    -------
    bool
        True if dim_matrix @ exponents == target_dim within tolerance.
    """
    result_dim = dim_matrix @ exponents
    return np.allclose(result_dim, target_dim, atol=tol)


def filter_dimensionally_valid(
    candidate_exponents: list[np.ndarray],
    dim_matrix: np.ndarray,
    target_dim: np.ndarray,
    tol: float = 1e-10,
) -> list[np.ndarray]:
    """Filter a list of candidate monomials to keep only dimensionally valid ones.

    Parameters
    ----------
    candidate_exponents : list of np.ndarray
        Each array is shape (n_vars,) giving exponents for a monomial.
    dim_matrix : np.ndarray, shape (n_dims, n_vars)
        Dimensional matrix.
    target_dim : np.ndarray, shape (n_dims,)
        Target variable dimensions.
    tol : float
        Tolerance for checking.

    Returns
    -------
    list of np.ndarray
        Dimensionally consistent monomials only.
    """
    return [
        exp for exp in candidate_exponents
        if check_dimensional_consistency(exp, dim_matrix, target_dim, tol)
    ]
