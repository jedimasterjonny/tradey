"""Tests for pure math functions: SSE, Jacobian, Hessian, and allocation deviation."""

import numpy as np
import pytest

from tradey import Portfolio


@pytest.fixture
def simple_portfolio():
    """A synthetic 3-asset portfolio for math tests."""
    return Portfolio(
        allocations={"Fund A": 6000.0, "Fund B": 3000.0, "Fund C": 1000.0},
        allocation_targets={"Fund A": 60.0, "Fund B": 30.0, "Fund C": 10.0},
    )


class TestCalculateAllocationDeviation:
    def test_perfect_allocation(self):
        """When values exactly match target weights, deviation should be zero."""
        values = np.array([600.0, 300.0, 100.0])
        targets = np.array([0.6, 0.3, 0.1])
        deviation = Portfolio._calculate_allocation_deviation(values, targets)
        np.testing.assert_allclose(deviation, 0.0, atol=1e-10)

    def test_overweight_first_asset(self):
        """When first asset is overweight, its deviation should be positive."""
        values = np.array([800.0, 100.0, 100.0])
        targets = np.array([0.6, 0.3, 0.1])
        deviation = Portfolio._calculate_allocation_deviation(values, targets)
        # Fund A: 800 / (0.6 * 1000) - 1 = 1/3
        assert deviation[0] == pytest.approx(1 / 3)
        # Fund B: 100 / (0.3 * 1000) - 1 = -2/3
        assert deviation[1] == pytest.approx(-2 / 3)
        # Fund C: 100 / (0.1 * 1000) - 1 = 0
        assert deviation[2] == pytest.approx(0.0)

    def test_two_assets(self):
        """Basic two-asset case."""
        values = np.array([70.0, 30.0])
        targets = np.array([0.5, 0.5])
        deviation = Portfolio._calculate_allocation_deviation(values, targets)
        # 70 / (0.5 * 100) - 1 = 0.4
        assert deviation[0] == pytest.approx(0.4)
        # 30 / (0.5 * 100) - 1 = -0.4
        assert deviation[1] == pytest.approx(-0.4)


class TestCalculateSSE:
    def test_zero_allocation_returns_current_sse(self):
        """Zero additional allocation should return SSE of current portfolio."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([0.0, 0.0])
        indices = (0, 1)

        sse = Portfolio._calculate_sse(allocations, indices, current, targets)

        # Current is perfectly balanced, SSE should be 0
        assert sse == pytest.approx(0.0, abs=1e-10)

    def test_allocation_improves_sse(self):
        """Adding to an underweight asset should reduce SSE."""
        current = np.array([7000.0, 2000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])

        sse_before = Portfolio._calculate_sse(np.array([0.0]), (1,), current, targets)
        # Add 1000 to underweight Fund B
        sse_after = Portfolio._calculate_sse(np.array([1000.0]), (1,), current, targets)
        assert sse_after < sse_before

    def test_sse_is_nonnegative(self):
        """SSE should always be non-negative (sum of squares)."""
        current = np.array([5000.0, 3000.0, 2000.0])
        targets = np.array([0.5, 0.3, 0.2])
        allocations = np.array([100.0, 200.0])
        sse = Portfolio._calculate_sse(allocations, (0, 2), current, targets)
        assert sse >= 0

    def test_single_asset_allocation(self):
        """Allocating to a single asset index."""
        current = np.array([4000.0, 4000.0, 2000.0])
        targets = np.array([0.5, 0.3, 0.2])
        allocations = np.array([500.0])
        sse = Portfolio._calculate_sse(allocations, (2,), current, targets)
        assert isinstance(sse, float)
        assert sse >= 0


class TestCalculateJacobian:
    def test_gradient_shape(self):
        """Jacobian should have same length as allocations."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0])
        jac = Portfolio._calculate_jacobian(allocations, (0, 1), current, targets)
        assert jac.shape == (2,)

    def test_gradient_at_optimum_is_near_zero(self):
        """At a balanced allocation, gradient components should be small."""
        # Perfectly balanced portfolio
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        # Allocate proportionally to maintain balance
        allocations = np.array([600.0, 300.0, 100.0])
        jac = Portfolio._calculate_jacobian(allocations, (0, 1, 2), current, targets)
        # All gradients should be equal (not zero, but equal — at the optimum
        # of the constrained problem, gradients are equal, not zero)
        np.testing.assert_allclose(jac[0], jac[1], rtol=1e-6)
        np.testing.assert_allclose(jac[1], jac[2], rtol=1e-6)

    def test_gradient_direction(self):
        """Gradient should point away from underweight assets."""
        current = np.array([7000.0, 2000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([0.0, 0.0])
        jac = Portfolio._calculate_jacobian(allocations, (0, 1), current, targets)
        # Fund A is overweight, Fund B is underweight
        # Gradient for A should be > gradient for B (adding to A costs more)
        assert jac[0] > jac[1]


class TestCalculateHessian:
    def test_hessian_shape(self):
        """Hessian should be n x n where n = len(allocations)."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0])
        hess = Portfolio._calculate_hessian(allocations, (0, 1), current, targets)
        assert hess.shape == (2, 2)

    def test_hessian_is_diagonal(self):
        """The Hessian implementation returns a diagonal matrix."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0, 50.0])
        hess = Portfolio._calculate_hessian(allocations, (0, 1, 2), current, targets)
        # Off-diagonal elements should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert hess[i, j] == pytest.approx(0.0)

    def test_hessian_positive_definite(self):
        """Hessian diagonal entries should be positive (convex problem)."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0])
        hess = Portfolio._calculate_hessian(allocations, (0, 1), current, targets)
        assert hess[0, 0] > 0
        assert hess[1, 1] > 0

    def test_smaller_target_has_larger_hessian(self):
        """Assets with smaller target weight should have larger Hessian entries."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([0.0, 0.0, 0.0])
        hess = Portfolio._calculate_hessian(allocations, (0, 1, 2), current, targets)
        # 0.1 target → largest Hessian, 0.6 target → smallest
        assert hess[2, 2] > hess[1, 1] > hess[0, 0]
