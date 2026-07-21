"""Pure allocation-optimization math.

Module-level functions only: no class, no ``self``, no I/O. Everything operates
on numpy arrays so the solver can be swapped without touching the data model.
"""

import itertools
import math
from functools import partial
from typing import cast

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def calculate_allocation_deviation(
    values: np.ndarray, target_weights: np.ndarray
) -> np.ndarray:
    total = cast(float, np.sum(values))
    # Avoid division by zero if target is 0 (though users should avoid 0 targets)
    deviation: np.ndarray = (values / (target_weights * total)) - 1
    return deviation


def sse(
    allocations: np.ndarray,
    model_indices: tuple[int, ...],
    current_vals: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    new_values = current_vals.copy()
    indices = np.array(model_indices, dtype=int)
    new_values[indices] += allocations

    new_weights = calculate_allocation_deviation(new_values, target_weights)
    return cast(float, np.sum(new_weights**2))


def jacobian(
    allocations: np.ndarray,
    model_indices: tuple[int, ...],
    current_vals: np.ndarray,
    target_weights: np.ndarray,
) -> np.ndarray:
    total_value = cast(float, np.sum(current_vals)) + cast(float, np.sum(allocations))
    indices = np.array(model_indices, dtype=int)

    active_current = current_vals[indices]
    active_targets = target_weights[indices]
    # Active allocations correspond directly to 'allocations' argument

    active_final_values = active_current + allocations

    denom: np.ndarray = active_targets * total_value
    weights: np.ndarray = (active_final_values / denom) - 1

    grad: np.ndarray = 2 * weights * (1 / denom)
    return grad


def hessian(
    allocations: np.ndarray,
    model_indices: tuple[int, ...],
    current_vals: np.ndarray,
    target_weights: np.ndarray,
) -> np.ndarray:
    total_value = cast(float, np.sum(current_vals)) + cast(float, np.sum(allocations))
    indices = np.array(model_indices, dtype=int)
    active_targets = target_weights[indices]

    denom: np.ndarray = active_targets * total_value
    diag_values: np.ndarray = 2 * (1 / denom) ** 2

    return np.diag(diag_values)


def _optimize_combination(
    combination: tuple[int, ...],
    additional_investment: float,
    current_vals: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[int, ...], float, np.ndarray, bool]:
    """Optimize a single asset combination."""
    # The number of assets is exactly the size of this combination.
    n_assets = len(combination)

    # Constraint: Sum of allocations must equal additional_investment
    constraint_matrix = np.ones((1, n_assets))
    linear_constraint = LinearConstraint(
        constraint_matrix, additional_investment, additional_investment
    )

    # Bounds: Non-negative allocations
    bounds = Bounds(lb=0.0, ub=additional_investment)

    # Initial guess: Even distribution
    initial_guess = [additional_investment / n_assets] * n_assets

    result = minimize(
        fun=sse,
        x0=initial_guess,
        args=(combination, current_vals, targets),
        method="trust-constr",
        bounds=bounds,
        constraints=[linear_constraint],
        jac=jacobian,
        hess=hessian,
    )

    return combination, float(result.fun), result.x, bool(result.success)


def optimize(
    current_vals: np.ndarray,
    targets: np.ndarray,
    additional_investment: float,
    n_assets: int,
    eligible_indices: list[int],
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Find the optimal allocation of additional funds across eligible assets.

    Returns ``(best_combination, best_allocations, min_sse)``. A plain tuple is
    returned rather than ``OptimizationResult`` so this module never imports the
    data model (keeps the dependency edge one-directional).
    """
    if n_assets > len(eligible_indices):
        n_assets = len(eligible_indices)

    min_overall_sse = float("inf")
    best_combination = None
    best_allocations = None

    combinations = list(itertools.combinations(eligible_indices, n_assets))

    func = partial(
        _optimize_combination,
        additional_investment=additional_investment,
        current_vals=current_vals,
        targets=targets,
    )

    # The combination count is tiny for a personal-scale portfolio, so a serial
    # map is used instead of a process pool (no pickling constraints).
    results = list(map(func, combinations))

    # Find best result, considering only combinations that converged.
    # Non-converged results may carry an `x` that violates the budget or
    # non-negativity constraints, so they must never be treated as valid.
    for combination, result_fun, result_x, success in results:
        if not success:
            continue
        if result_fun < min_overall_sse:
            min_overall_sse = result_fun
            best_combination = combination
            best_allocations = result_x

    if best_combination is None or best_allocations is None:
        raise ValueError(
            "optimization failed: no asset combination converged to a valid allocation"
        )

    # Post-validate the winning allocation before trusting it.
    best_allocations = np.asarray(best_allocations, dtype=float)
    alloc_sum = float(np.sum(best_allocations))
    if not math.isclose(alloc_sum, additional_investment, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            "optimization produced an invalid allocation: amounts sum to "
            f"{alloc_sum:.2f}, expected {additional_investment:.2f}"
        )
    if np.any(best_allocations < -1e-9):
        raise ValueError(
            "optimization produced an invalid allocation: a negative amount "
            f"was suggested ({best_allocations.tolist()})"
        )
    # Clip negligible negatives (within tolerance) to exactly zero.
    best_allocations = np.where(best_allocations < 0.0, 0.0, best_allocations)

    return best_combination, best_allocations, min_overall_sse
