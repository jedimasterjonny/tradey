"""Tests for pure math functions: SSE, Jacobian, Hessian, and allocation deviation."""

import numpy as np
import pytest

from tradey.optimizer import (
    calculate_allocation_deviation,
    hessian,
    jacobian,
)
from tradey.optimizer import sse as calculate_sse


class TestCalculateAllocationDeviation:
    @pytest.mark.parametrize(
        ("values", "targets", "expected"),
        [
            # Values exactly proportional to targets → zero deviation everywhere.
            ([600.0, 300.0, 100.0], [0.6, 0.3, 0.1], [0.0, 0.0, 0.0]),
            # First asset overweight. total=1000:
            #   800/(0.6*1000)-1 = +1/3, 100/(0.3*1000)-1 = -2/3, 100/(0.1*1000)-1 = 0
            ([800.0, 100.0, 100.0], [0.6, 0.3, 0.1], [1 / 3, -2 / 3, 0.0]),
            # Two-asset case. total=100:
            #   70/(0.5*100)-1 = +0.4, 30/(0.5*100)-1 = -0.4
            ([70.0, 30.0], [0.5, 0.5], [0.4, -0.4]),
        ],
    )
    def test_deviation(self, values, targets, expected):
        """Deviation is value/(target*total) - 1, computed component-wise."""
        deviation = calculate_allocation_deviation(np.array(values), np.array(targets))
        np.testing.assert_allclose(deviation, expected, atol=1e-10)


class TestCalculateSSE:
    def test_balanced_portfolio_has_zero_sse(self):
        """A perfectly balanced portfolio with no new cash has zero SSE."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        result = calculate_sse(np.array([0.0, 0.0]), (0, 1), current, targets)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_allocation_to_underweight_reduces_sse(self):
        """Adding to an underweight asset should reduce SSE."""
        current = np.array([7000.0, 2000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        sse_before = calculate_sse(np.array([0.0]), (1,), current, targets)
        sse_after = calculate_sse(np.array([1000.0]), (1,), current, targets)
        assert sse_after < sse_before

    @pytest.mark.parametrize(
        ("allocations", "indices", "current", "targets", "expected"),
        [
            # [5000,3000,2000] + 100→idx0, 200→idx2 = [5100,3000,2200], total 10300.
            # dev = [5100/5150-1, 3000/3090-1, 2200/2060-1]
            #     = [-0.00970874, -0.02912621, 0.06796117]
            # sse = sum(dev^2) = 0.005561315863889146
            (
                [100.0, 200.0],
                (0, 2),
                [5000.0, 3000.0, 2000.0],
                [0.5, 0.3, 0.2],
                0.005561315863889146,
            ),
            # [4000,4000,2000] + 500→idx2 = [4000,4000,2500], total 10500.
            # dev = [4000/5250-1, 4000/3150-1, 2500/2100-1]
            #     = [-0.23809524, 0.26984127, 0.19047619]
            # sse = sum(dev^2) = 0.16578483245149908
            (
                [500.0],
                (2,),
                [4000.0, 4000.0, 2000.0],
                [0.5, 0.3, 0.2],
                0.16578483245149908,
            ),
        ],
    )
    def test_sse_exact_value(self, allocations, indices, current, targets, expected):
        """SSE equals the hand-computed sum of squared deviations (not just >=0)."""
        result = calculate_sse(
            np.array(allocations), indices, np.array(current), np.array(targets)
        )
        assert result == pytest.approx(expected)


class TestCalculateJacobian:
    def test_gradient_shape(self):
        """Jacobian should have same length as allocations."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0])
        jac = jacobian(allocations, (0, 1), current, targets)
        assert jac.shape == (2,)

    def test_gradient_at_optimum_is_near_zero(self):
        """At a balanced allocation, gradient components should be small."""
        # Perfectly balanced portfolio
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        # Allocate proportionally to maintain balance
        allocations = np.array([600.0, 300.0, 100.0])
        jac = jacobian(allocations, (0, 1, 2), current, targets)
        # All gradients should be equal (not zero, but equal — at the optimum
        # of the constrained problem, gradients are equal, not zero)
        np.testing.assert_allclose(jac[0], jac[1], rtol=1e-6)
        np.testing.assert_allclose(jac[1], jac[2], rtol=1e-6)

    def test_gradient_direction(self):
        """Gradient should point away from underweight assets."""
        current = np.array([7000.0, 2000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([0.0, 0.0])
        jac = jacobian(allocations, (0, 1), current, targets)
        # Fund A is overweight, Fund B is underweight
        # Gradient for A should be > gradient for B (adding to A costs more)
        assert jac[0] > jac[1]


class TestJacobianConstraintParallelError:
    """Pin the deliberate jacobian/true-gradient discrepancy (audit correctness #11).

    ``jacobian`` differentiates SSE while treating the portfolio total as a
    constant. Differentiating SSE properly (see ``sse``, whose deviation uses
    ``total = sum(all values)``) adds a term ``-C`` that is *identical for every
    component*: it depends only on the shared portfolio total, not on which
    asset is being differentiated. Hence ``analytic - numeric`` is a vector with
    (near-)constant components — i.e. parallel to the all-ones vector.

    Why this is safe: the equality constraint ``sum(allocations) == investment``
    means feasible steps lie in the ``sum == 0`` subspace, which is orthogonal to
    the all-ones direction — so a constant offset in the gradient does not change
    the constrained optimum. It is harmless *only while that constraint exists*.
    This test fails if a change makes the error vary across components (a
    non-constant error would no longer be projected out and would bias the solve).
    """

    def test_analytic_minus_numeric_is_constant_across_components(self):
        current = np.array([7000.0, 2000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        combination = (0, 1, 2)
        # An off-optimum, non-proportional point so both gradients are non-trivial.
        allocations = np.array([100.0, 800.0, 100.0])

        analytic = jacobian(allocations, combination, current, targets)

        # Central-difference numeric gradient of sse (step is tiny vs the
        # ~100-800 allocation scale, so truncation error stays negligible).
        step = 0.1
        numeric = np.empty_like(allocations)
        for k in range(len(allocations)):
            plus = allocations.copy()
            plus[k] += step
            minus = allocations.copy()
            minus[k] -= step
            numeric[k] = (
                calculate_sse(plus, combination, current, targets)
                - calculate_sse(minus, combination, current, targets)
            ) / (2 * step)

        diff = analytic - numeric
        # The discrepancy is a genuine, non-zero constant offset (guards against
        # a degenerate pass where both gradients happen to be ~0).
        assert abs(diff.mean()) > 1e-7
        # ...and every component barely deviates from that shared mean.
        assert np.max(np.abs(diff - diff.mean())) < 1e-9


class TestCalculateHessian:
    def test_hessian_shape(self):
        """Hessian should be n x n where n = len(allocations)."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0])
        hess = hessian(allocations, (0, 1), current, targets)
        assert hess.shape == (2, 2)

    def test_hessian_is_diagonal(self):
        """The Hessian implementation returns a diagonal matrix."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([100.0, 200.0, 50.0])
        hess = hessian(allocations, (0, 1, 2), current, targets)
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
        hess = hessian(allocations, (0, 1), current, targets)
        assert hess[0, 0] > 0
        assert hess[1, 1] > 0

    def test_smaller_target_has_larger_hessian(self):
        """Assets with smaller target weight should have larger Hessian entries."""
        current = np.array([6000.0, 3000.0, 1000.0])
        targets = np.array([0.6, 0.3, 0.1])
        allocations = np.array([0.0, 0.0, 0.0])
        hess = hessian(allocations, (0, 1, 2), current, targets)
        # 0.1 target → largest Hessian, 0.6 target → smallest
        assert hess[2, 2] > hess[1, 1] > hess[0, 0]
