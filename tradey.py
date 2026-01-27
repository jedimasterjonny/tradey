from scipy.optimize import minimize, LinearConstraint, Bounds
from prettytable import PrettyTable
import argparse
import multiprocessing
from functools import partial
import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np


class OptimizationResult(NamedTuple):
    """Result of the portfolio allocation optimization."""

    best_combination: Tuple[int, ...]
    best_allocations: np.ndarray
    min_sse: float


@dataclass
class Portfolio:
    """Manages portfolio data and optimization operations."""

    allocations: Dict[str, float]
    allocation_targets: Dict[str, float]
    category_totals: Dict[str, float]
    investment_names: List[str] = field(init=False)

    def __post_init__(self):
        """Initialize derived attributes."""
        self.investment_names = list(self.allocations.keys())

    @property
    def current_values(self) -> np.ndarray:
        """Get current investment values as a numpy array."""
        return np.array(list(self.allocations.values()))

    @property
    def target_weights(self) -> np.ndarray:
        """Get target weights as a numpy array."""

        return np.array(list(self.allocation_targets.values())) / 100.0

    @classmethod
    def from_csv(cls, filepath: Path) -> "Portfolio":
        """Parse asset allocation CSV and return a Portfolio instance."""
        filepath = Path(filepath)
        current_category = ""
        category_totals = {}
        allocations = {}
        allocation_targets = {}

        def parse_currency(value_str: str) -> float:
            """Parse currency string by removing commas and converting to float."""
            return float(value_str.replace(",", ""))

        with filepath.open("r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')

            for row_id, row in enumerate(reader):
                # Skip header row
                if row_id == 0:
                    continue
                # Skip total row (row 1, index 1)
                elif row_id == 1:
                    continue

                category = row[1]
                # Process category headers (Equity, Bonds)
                if category in ["Equity", "Bonds"] and category != current_category:
                    current_category = category
                    category_totals[current_category] = float(row[5])
                    continue

                # Process allocation rows within current category
                if category == current_category and row[4] == "":
                    asset_name = row[2]
                    allocations[asset_name] = parse_currency(row[10])

                    # Calculate target allocation
                    target_percentage = parse_currency(row[5])
                    category_weight = category_totals[category] / 100
                    allocation_targets[asset_name] = target_percentage * category_weight

        return cls(
            allocations=allocations,
            allocation_targets=allocation_targets,
            category_totals=category_totals,
        )

    def optimize(
        self,
        additional_investment: float,
        n_assets: int = 2,
        excluded_assets: Optional[List[str]] = None,
    ) -> OptimizationResult:
        """
        Find optimal allocation for additional funds using parallel processing.

        Args:
            additional_investment: Amount of new valid to allocate.
            n_assets: Number of assets to split the investment across.
            excluded_assets: List of asset names to exclude from optimization.
        """
        current_vals = self.current_values
        targets = self.target_weights

        if excluded_assets is None:
            excluded_assets = []

        # Find indices of eligible investments (those not excluded)
        eligible_indices = [
            i
            for i, name in enumerate(self.investment_names)
            if name not in excluded_assets
        ]

        if not eligible_indices:
            raise ValueError("No eligible assets available for optimization.")

        num_eligible = len(eligible_indices)
        if n_assets is None or n_assets > num_eligible:
            n_assets = num_eligible

        investment_indices = eligible_indices

        min_overall_sse = float("inf")
        best_combination = None
        best_allocations = None

        # Prepare arguments for parallel execution
        combinations = list(itertools.combinations(investment_indices, n_assets))

        # Helper function arguments
        func = partial(
            self._optimize_combination,
            additional_investment=additional_investment,
            current_vals=current_vals,
            targets=targets,
            n_assets=n_assets,
        )

        # Use multiprocessing to speed up optimization
        # Default to number of CPU cores
        with multiprocessing.Pool() as pool:
            results = pool.map(func, combinations)

        # Find best result
        for combination, result_fun, result_x in results:
            if result_fun < min_overall_sse:
                min_overall_sse = result_fun
                best_combination = combination
                best_allocations = result_x

        return OptimizationResult(
            best_combination=best_combination,  # type: ignore
            best_allocations=best_allocations,
            min_sse=min_overall_sse,
        )

    @classmethod
    def _optimize_combination(
        cls,
        combination: Tuple[int, ...],
        additional_investment: float,
        current_vals: np.ndarray,
        targets: np.ndarray,
        n_assets: int,
    ) -> Tuple[Tuple[int, ...], float, np.ndarray]:
        """Helper method to optimize a single combination (static for pickling)."""

        # Constraint: Sum of allocations must equal additional_investment
        # We assume n_assets matches len(combination)
        constraint_matrix = np.ones((1, n_assets))
        linear_constraint = LinearConstraint(
            constraint_matrix, [additional_investment], [additional_investment]
        )

        # Bounds: Non-negative allocations
        bounds = Bounds(np.zeros(n_assets), np.full(n_assets, additional_investment))

        # Initial guess: Even distribution
        initial_guess = [additional_investment / n_assets] * n_assets

        result = minimize(
            fun=cls._calculate_sse,
            x0=initial_guess,
            args=(combination, current_vals, targets),
            method="trust-constr",
            bounds=bounds,
            constraints=[linear_constraint],
            jac=cls._calculate_jacobian,
            hess=cls._calculate_hessian,
        )

        return combination, result.fun, result.x

    @staticmethod
    def _calculate_allocation_deviation(
        values: np.ndarray, target_weights: np.ndarray
    ) -> np.ndarray:
        total = np.sum(values)
        # Avoid division by zero if target is 0 (though users should avoid 0 targets)
        deviation = (values / (target_weights * total)) - 1
        return deviation

    @classmethod
    def _calculate_sse(
        cls,
        allocations: np.ndarray,
        model_indices: Tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> float:
        new_values = current_vals.copy()
        indices = np.array(model_indices, dtype=int)
        new_values[indices] += allocations

        new_weights = cls._calculate_allocation_deviation(new_values, target_weights)
        return np.sum(new_weights**2)

    @staticmethod
    def _calculate_jacobian(
        allocations: np.ndarray,
        model_indices: Tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        total_value = np.sum(current_vals) + np.sum(allocations)
        indices = np.array(model_indices, dtype=int)

        active_current = current_vals[indices]
        active_targets = target_weights[indices]
        # Active allocations correspond directly to 'allocations' argument

        active_final_values = active_current + allocations

        denom = active_targets * total_value
        weights = (active_final_values / denom) - 1

        grad = 2 * weights * (1 / denom)
        return grad

    @staticmethod
    def _calculate_hessian(
        allocations: np.ndarray,
        model_indices: Tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        total_value = np.sum(current_vals) + np.sum(allocations)
        indices = np.array(model_indices, dtype=int)
        active_targets = target_weights[indices]

        denom = active_targets * total_value
        diag_values = 2 * (1 / denom) ** 2

        return np.diag(diag_values)

    def print_summary(
        self, optimization_result: OptimizationResult, additional_investment: float
    ):
        """Print the results of the optimization."""
        best_combination = optimization_result.best_combination
        best_allocations = optimization_result.best_allocations

        print("--- Optimal Allocation Found ---")
        param_names = [self.investment_names[i] for i in best_combination]
        print(f"Invest in: {param_names}")

        for idx, amount in zip(best_combination, best_allocations):
            name = self.investment_names[idx]
            percent = (amount / additional_investment) * 100
            print(f"Optimal allocation: £{amount:,.2f} to '{name}' ({percent:.2f}%)")

        print("\n")
        print(f"SSE: {optimization_result.min_sse:.2e}\n")

        # Portfolio Weights Table
        final_values = self.current_values.copy()

        # Apply optimal allocations
        indices = np.array(best_combination, dtype=int)
        final_values[indices] += best_allocations

        targets = self.target_weights
        final_dev = self._calculate_allocation_deviation(final_values, targets)
        old_dev = self._calculate_allocation_deviation(self.current_values, targets)

        print("--- Portfolio Weights ---")
        output = PrettyTable()
        output.field_names = ["Investment", "Target", "Old Deviance", "New Deviance"]
        output.align = "r"
        output.align["Investment"] = "l"

        for i, name in enumerate(self.investment_names):
            output.add_row(
                [
                    name,
                    f"{targets[i]:.1%}",
                    f"{old_dev[i]:.1%}",
                    f"{final_dev[i]:.1%}",
                ]
            )
        print(output)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate optimal asset allocation.")
    parser.add_argument(
        "filepath",
        help="Path to the asset allocation CSV file",
    )
    parser.add_argument(
        "additional_investment",
        type=float,
        help="Amount of additional cash to invest",
    )
    parser.add_argument(
        "--n-assets",
        "-n",
        type=int,
        default=2,
        help="Number of assets to distribute investment across (default: 2)",
    )
    parser.add_argument(
        "--exclude",
        "-x",
        nargs="+",
        default=[],
        help="List of asset names to exclude from optimization",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        portfolio = Portfolio.from_csv(args.filepath)
        result = portfolio.optimize(
            additional_investment=args.additional_investment,
            n_assets=args.n_assets,
            excluded_assets=args.exclude,
        )
        portfolio.print_summary(result, args.additional_investment)

    except FileNotFoundError:
        print(f"Error: Could not find file at {args.filepath}")
        exit(1)
    except Exception as e:
        print(f"Error processing portfolio: {e}")
        exit(1)


if __name__ == "__main__":
    main()
