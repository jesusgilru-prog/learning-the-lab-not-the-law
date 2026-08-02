"""Tests for Buckingham Pi dimensional analysis."""

import numpy as np
import pytest

from sr_engine.buckingham_pi import (
    find_pi_groups,
    n_pi_groups,
    check_dimensional_consistency,
    filter_dimensionally_valid,
)


class TestPendulum:
    """Classic pendulum: period T = 2π√(L/g).

    Variables: T (period), L (length), g (acceleration).
    Base dimensions: [L, M, T]

    Dimensional matrix:
        T   L   g
    L:  0   1   1
    M:  0   0   0
    T:  1   0  -2
    """

    @pytest.fixture
    def dim_matrix(self):
        return np.array([
            [0, 1,  1],  # L
            [0, 0,  0],  # M
            [1, 0, -2],  # T
        ])

    def test_number_of_pi_groups(self, dim_matrix):
        # 3 variables, rank 2 -> 1 Pi group
        assert n_pi_groups(dim_matrix) == 1

    def test_pi_group_structure(self, dim_matrix):
        groups = find_pi_groups(dim_matrix, ['T', 'L', 'g'])
        assert len(groups) == 1
        group = groups[0]
        # Pi = T^a * L^b * g^c must be dimensionless
        # Solution: T^2 * g / L (or any scalar multiple)
        # Check that 2*a_T = -a_g and a_L = -a_g
        assert 'T' in group
        assert 'g' in group
        # The ratio T_exp / g_exp should be 2 (T^2 * g)
        ratio = group['T'] / group['g']
        assert abs(ratio - 2.0) < 1e-10 or abs(ratio + 2.0) < 1e-10


class TestReynolds:
    """Reynolds number: Re = ρvL/μ.

    Variables: ρ (density), v (velocity), L (length), μ (dynamic viscosity).
    Base dimensions: [L, M, T]

    Dimensional matrix:
         ρ    v    L    μ
    L:  -3    1    1   -1
    M:   1    0    0    1
    T:   0   -1    0   -1
    """

    @pytest.fixture
    def dim_matrix(self):
        return np.array([
            [-3,  1, 1, -1],  # L
            [ 1,  0, 0,  1],  # M
            [ 0, -1, 0, -1],  # T
        ])

    def test_number_of_pi_groups(self, dim_matrix):
        # 4 variables, rank 3 -> 1 Pi group
        assert n_pi_groups(dim_matrix) == 1

    def test_pi_group_is_reynolds(self, dim_matrix):
        groups = find_pi_groups(dim_matrix, ['rho', 'v', 'L', 'mu'])
        assert len(groups) == 1
        group = groups[0]
        # Re = ρ^1 * v^1 * L^1 * μ^-1
        # All exponents should have same magnitude
        vals = list(group.values())
        # mu should have opposite sign to the rest
        assert 'mu' in group
        assert 'rho' in group
        rho_sign = np.sign(group['rho'])
        mu_sign = np.sign(group['mu'])
        assert rho_sign != mu_sign


class TestDragForce:
    """Drag force: F = f(ρ, v, L, μ).

    Variables: F, ρ, v, L, μ
    Base dimensions: [L, M, T]

    Dimensional matrix:
          F    ρ    v    L    μ
    L:    1   -3    1    1   -1
    M:    1    1    0    0    1
    T:   -2    0   -1    0   -1
    """

    @pytest.fixture
    def dim_matrix(self):
        return np.array([
            [ 1, -3,  1, 1, -1],  # L
            [ 1,  1,  0, 0,  1],  # M
            [-2,  0, -1, 0, -1],  # T
        ])

    def test_number_of_pi_groups(self, dim_matrix):
        # 5 variables, rank 3 -> 2 Pi groups
        assert n_pi_groups(dim_matrix) == 2

    def test_pi_groups_exist(self, dim_matrix):
        groups = find_pi_groups(dim_matrix, ['F', 'rho', 'v', 'L', 'mu'])
        assert len(groups) == 2


class TestDimensionalConsistency:
    """Test dimensional consistency checking."""

    def test_consistent_monomial(self):
        # y has dimension L^1, variables x0 (L^1) and x1 (L^0 T^1)
        dim_matrix = np.array([
            [1, 0],  # L
            [0, 1],  # T
        ])
        target_dim = np.array([1, 0])  # y ~ L^1
        exponents = np.array([1.0, 0.0])  # x0^1 has dim L^1
        assert check_dimensional_consistency(exponents, dim_matrix, target_dim)

    def test_inconsistent_monomial(self):
        dim_matrix = np.array([
            [1, 0],  # L
            [0, 1],  # T
        ])
        target_dim = np.array([1, 0])  # y ~ L^1
        exponents = np.array([0.0, 1.0])  # x1^1 has dim T^1, not L^1
        assert not check_dimensional_consistency(exponents, dim_matrix, target_dim)

    def test_filter_valid(self):
        dim_matrix = np.array([
            [1, 0],  # L
            [0, 1],  # T
        ])
        target_dim = np.array([2, -1])  # y ~ L^2 T^-1

        candidates = [
            np.array([2.0, -1.0]),  # L^2 T^-1 ✓
            np.array([1.0, 1.0]),   # L^1 T^1 ✗
            np.array([0.0, 0.0]),   # dimensionless ✗
            np.array([2.0, -1.0]),  # duplicate valid ✓
        ]
        valid = filter_dimensionally_valid(candidates, dim_matrix, target_dim)
        assert len(valid) == 2


class TestEdgeCases:
    """Edge cases for dimensional analysis."""

    def test_single_variable(self):
        dim_matrix = np.array([[1]])  # single var with dim L^1
        assert n_pi_groups(dim_matrix) == 0  # no dimensionless groups

    def test_all_dimensionless(self):
        dim_matrix = np.array([[0, 0, 0]])  # all dimensionless
        groups = find_pi_groups(dim_matrix, ['a', 'b', 'c'])
        assert len(groups) == 3  # each variable is its own Pi group

    def test_variable_names_mismatch(self):
        dim_matrix = np.array([[1, 0]])
        with pytest.raises(ValueError):
            find_pi_groups(dim_matrix, ['a', 'b', 'c'])  # 3 names, 2 cols
