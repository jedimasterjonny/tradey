"""Core data types: the Portfolio dataclass and the optimization result."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from tradey import optimizer


class OptimizationResult(NamedTuple):
    """Result of the portfolio allocation optimization."""

    best_combination: tuple[int, ...]
    best_allocations: np.ndarray
    min_sse: float


@dataclass
class Portfolio:
    """A rebalanceable portfolio: holdings, target weights and base currency."""

    allocations: dict[str, float]
    allocation_targets: dict[str, float]
    base_currency: str = "GBP"
    investment_names: list[str] = field(init=False)
    current_values: np.ndarray = field(init=False)
    target_weights: np.ndarray = field(init=False)

    def __post_init__(self):
        """Validate invariants and precompute derived arrays once."""
        # Every sub-category must have a positive target weight, otherwise the
        # deviation calculation divides by zero and poisons the optimizer with
        # nan/inf. (Guarded here so directly-constructed portfolios are safe too.)
        zero_target_names = [
            name for name, target in self.allocation_targets.items() if target <= 0
        ]
        if zero_target_names:
            raise ValueError(
                "the following sub-categories have no target weight (weight must "
                f"be greater than 0): {zero_target_names}"
            )

        # The portfolio must have a positive total value, otherwise every
        # deviation is nan (division by a zero total).
        total_value = sum(self.allocations.values())
        if total_value <= 0:
            raise ValueError(
                "portfolio has no value (total holdings are zero); nothing to rebalance"
            )

        # Precompute derived arrays once. Correctness relies on allocations and
        # allocation_targets sharing identical key order (the loader builds both
        # in the same loop; direct construction must do likewise).
        self.investment_names = list(self.allocations.keys())
        self.current_values = np.array(list(self.allocations.values()))
        self.target_weights = np.array(list(self.allocation_targets.values())) / 100.0

    @classmethod
    def from_portfolio_file(
        cls,
        filepath,
        taxonomy_name: str = "Asset Allocation",
    ) -> Portfolio:
        """Parse a .portfolio file and return a Portfolio instance.

        Thin backward-compatible wrapper around :func:`tradey.loader.load_portfolio`.
        """
        from tradey.loader import load_portfolio

        return load_portfolio(filepath, taxonomy_name=taxonomy_name)

    def optimize(
        self,
        additional_investment: float,
        n_assets: int = 2,
        excluded_assets: list[str] | None = None,
    ) -> OptimizationResult:
        """Find the optimal allocation for additional funds.

        Args:
            additional_investment: Amount of new cash to allocate.
            n_assets: Number of assets to split the investment across.
            excluded_assets: List of asset names to exclude from optimization.
        """
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

        best_combination, best_allocations, min_sse = optimizer.optimize(
            self.current_values,
            self.target_weights,
            additional_investment,
            n_assets,
            eligible_indices,
        )

        return OptimizationResult(
            best_combination=best_combination,
            best_allocations=best_allocations,
            min_sse=min_sse,
        )
