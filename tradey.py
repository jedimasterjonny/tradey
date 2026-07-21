import argparse
import itertools
import json
import math
import multiprocessing
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from functools import partial
from http.client import HTTPResponse
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
from google.protobuf.message import DecodeError
from prettytable import PrettyTable
from scipy.optimize import Bounds, LinearConstraint, minimize

from proto.client_pb2 import PClient

# Precision constants used by Portfolio Performance
_SHARE_PRECISION = 1e8
_PRICE_PRECISION = 1e8

_SIGNATURE = b"PPPBV1"

# Fixed currency conversion factors (not available via exchange rate APIs)
_FIXED_CURRENCY_FACTORS = {
    "GBX": {"GBP": 0.01},  # pence to pounds
}


def _fetch_exchange_rates(base_currency: str, currencies: set[str]) -> dict[str, float]:
    """Fetch exchange rates from the Frankfurter API (ECB data).

    Returns a dict mapping each requested currency to its conversion factor
    to the base currency.
    """
    if not currencies:
        return {}

    symbols = ",".join(sorted(currencies))
    url = (
        f"https://api.frankfurter.dev/v1/latest?base={base_currency}&symbols={symbols}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "tradey/1.0"})
    resp = cast(HTTPResponse, urllib.request.urlopen(req, timeout=10))
    data = cast(dict[str, dict[str, float]], json.loads(resp.read()))
    resp.close()

    # API returns rates as base→foreign, we need foreign→base (i.e. 1/rate)
    return {currency: 1.0 / rate for currency, rate in data["rates"].items()}


def _read_client(filepath: Path) -> PClient:
    """Read and parse a Portfolio Performance .portfolio file."""
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            raw = zf.read("data.portfolio")
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a valid Portfolio Performance file: {e}") from e
    except KeyError as e:
        raise ValueError(
            "not a valid Portfolio Performance file: missing 'data.portfolio' member"
        ) from e

    signature = raw[: len(_SIGNATURE)]
    if signature != _SIGNATURE:
        raise ValueError(
            f"Not a Portfolio Performance protobuf file "
            f"(expected signature {_SIGNATURE!r}, got {signature!r})"
        )

    client = PClient()
    try:
        _ = client.ParseFromString(raw[len(_SIGNATURE) :])
    except DecodeError as e:
        raise ValueError(f"not a valid Portfolio Performance file: {e}") from e
    return client


def _compute_shares(client: PClient) -> dict[str, int]:
    """Calculate shares held per security from the transaction history."""
    sec_shares: dict[str, int] = {}
    for t in client.transactions:
        if not t.HasField("security"):
            continue
        sid = t.security
        shares = t.shares if t.HasField("shares") else 0
        if t.type in (0, 2):  # PURCHASE, INBOUND_DELIVERY
            sec_shares[sid] = sec_shares.get(sid, 0) + shares
        elif t.type in (1, 3):  # SALE, OUTBOUND_DELIVERY
            sec_shares[sid] = sec_shares.get(sid, 0) - shares
    return sec_shares


def _resolve_currency_factors(
    base_currency: str,
    assigned_currencies: set[str],
) -> dict[str, float]:
    """Map each currency to its conversion factor to the base currency."""
    factors: dict[str, float] = {base_currency: 1.0}
    for currency in assigned_currencies:
        fixed = _FIXED_CURRENCY_FACTORS.get(currency, {})
        if base_currency in fixed:
            factors[currency] = fixed[base_currency]

    need_fetch = assigned_currencies - set(factors)
    if need_fetch:
        factors.update(_fetch_exchange_rates(base_currency, need_fetch))

    return factors


class OptimizationResult(NamedTuple):
    """Result of the portfolio allocation optimization."""

    best_combination: tuple[int, ...]
    best_allocations: np.ndarray
    min_sse: float


@dataclass
class Portfolio:
    """Manages portfolio data and optimization operations."""

    allocations: dict[str, float]
    allocation_targets: dict[str, float]
    investment_names: list[str] = field(init=False)

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
    def from_portfolio_file(
        cls,
        filepath: Path,
        taxonomy_name: str = "Asset Allocation",
    ) -> Portfolio:
        """Parse a .portfolio file and return a Portfolio instance.

        Args:
            filepath: Path to the .portfolio file (zip containing protobuf data).
            taxonomy_name: Name of the taxonomy to use for allocation targets.
        """
        client = _read_client(Path(filepath))

        base_currency: str = client.baseCurrency or "GBP"
        sec_map = {s.uuid: s for s in client.securities}
        sec_shares = _compute_shares(client)

        # Find the target taxonomy
        taxonomy = None
        for tax in client.taxonomies:
            if tax.name == taxonomy_name:
                taxonomy = tax
                break
        if taxonomy is None:
            available = [t.name for t in client.taxonomies]
            raise ValueError(
                f"Taxonomy '{taxonomy_name}' not found. Available: {available}"
            )

        # Build classification tree
        classifications = {c.id: c for c in taxonomy.classifications}

        # Find root (no parentId)
        root = None
        for c in taxonomy.classifications:
            if not c.HasField("parentId"):
                root = c
                break

        # Build children map
        children: dict[str, list[str]] = {}
        for c in taxonomy.classifications:
            if c.HasField("parentId"):
                children.setdefault(c.parentId, []).append(c.id)

        if root is None:
            raise ValueError("Taxonomy has no root classification (missing parentId)")

        # Top-level categories are children of root (e.g. Equity, Bonds)
        top_level_ids = children.get(root.id, [])

        # Resolve currency conversion factors
        assigned_currencies: set[str] = set()
        for c in taxonomy.classifications:
            for a in c.assignments:
                sec = sec_map.get(a.investmentVehicle)
                if sec:
                    assigned_currencies.add(sec.currencyCode)
        currency_factors = _resolve_currency_factors(base_currency, assigned_currencies)

        def _security_value_base(uuid: str) -> float:
            """Calculate the current value of a security in base currency."""
            sec = sec_map.get(uuid)
            if sec is None or not sec.prices:
                return 0.0
            shares = sec_shares.get(uuid, 0) / _SHARE_PRECISION
            price = sec.prices[-1].close / _PRICE_PRECISION
            return shares * price * currency_factors[sec.currencyCode]

        # Process sub-categories (children of top-level categories)
        allocations: dict[str, float] = {}
        allocation_targets: dict[str, float] = {}

        for cat_id in top_level_ids:
            cat = classifications[cat_id]
            cat_weight = cat.weight / 10000.0  # as fraction
            sub_ids = children.get(cat_id, [])

            for sub_id in sub_ids:
                sub = classifications[sub_id]

                # Sum values of all securities assigned to this sub-category
                total_value = sum(
                    _security_value_base(a.investmentVehicle) for a in sub.assignments
                )
                allocations[sub.name] = total_value

                # Target = sub_weight * category_weight (both as fractions of parent)
                sub_weight_pct = (
                    sub.weight / 100.0
                )  # sub weight as percentage of category
                allocation_targets[sub.name] = sub_weight_pct * cat_weight

        # Fix #2: every included sub-category must have a positive target weight,
        # otherwise the deviation calculation divides by zero and poisons the
        # optimizer with nan/inf.
        zero_target_names = [
            name for name, target in allocation_targets.items() if target <= 0
        ]
        if zero_target_names:
            raise ValueError(
                "the following sub-categories have no target weight (weight must "
                f"be greater than 0): {zero_target_names}"
            )

        # Fix #2: the portfolio must have a positive total value, otherwise every
        # deviation is nan (division by a zero total).
        total_value = sum(allocations.values())
        if total_value <= 0:
            raise ValueError(
                "portfolio has no value (total holdings are zero); nothing to rebalance"
            )

        # Fix #3: normalize target weights so they sum to 1.0. Real taxonomies
        # often do not sum to 100% (unclassified cash, partial weights), which
        # would bias every deviation. allocation_targets are stored as
        # percentages (summing to 100 when complete).
        raw_sum = sum(allocation_targets.values()) / 100.0  # as a fraction
        if abs(raw_sum - 1.0) > 0.01:
            print(
                f"warning: taxonomy target weights sum to {raw_sum:.2%} "
                "(expected 100%); rescaling targets to sum to 1.0",
                file=sys.stderr,
            )
        for name in allocation_targets:
            allocation_targets[name] /= raw_sum

        return cls(
            allocations=allocations,
            allocation_targets=allocation_targets,
        )

    def optimize(
        self,
        additional_investment: float,
        n_assets: int = 2,
        excluded_assets: list[str] | None = None,
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

        if n_assets > len(eligible_indices):
            n_assets = len(eligible_indices)

        min_overall_sse = float("inf")
        best_combination = None
        best_allocations = None

        # Prepare arguments for parallel execution
        combinations = list(itertools.combinations(eligible_indices, n_assets))

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
                "optimization failed: no asset combination converged to a valid "
                "allocation"
            )

        # Post-validate the winning allocation before trusting it.
        best_allocations = np.asarray(best_allocations, dtype=float)
        alloc_sum = float(np.sum(best_allocations))
        if not math.isclose(
            alloc_sum, additional_investment, rel_tol=1e-6, abs_tol=1e-9
        ):
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

        return OptimizationResult(
            best_combination=best_combination,
            best_allocations=best_allocations,
            min_sse=min_overall_sse,
        )

    @classmethod
    def _optimize_combination(
        cls,
        combination: tuple[int, ...],
        additional_investment: float,
        current_vals: np.ndarray,
        targets: np.ndarray,
        n_assets: int,
    ) -> tuple[tuple[int, ...], float, np.ndarray, bool]:
        """Helper method to optimize a single combination (static for pickling)."""

        # Constraint: Sum of allocations must equal additional_investment
        # We assume n_assets matches len(combination)
        constraint_matrix = np.ones((1, n_assets))
        linear_constraint = LinearConstraint(
            constraint_matrix, additional_investment, additional_investment
        )

        # Bounds: Non-negative allocations
        bounds = Bounds(lb=0.0, ub=additional_investment)

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

        return combination, float(result.fun), result.x, bool(result.success)

    @staticmethod
    def _calculate_allocation_deviation(
        values: np.ndarray, target_weights: np.ndarray
    ) -> np.ndarray:
        total = cast(float, np.sum(values))
        # Avoid division by zero if target is 0 (though users should avoid 0 targets)
        deviation: np.ndarray = (values / (target_weights * total)) - 1
        return deviation

    @classmethod
    def _calculate_sse(
        cls,
        allocations: np.ndarray,
        model_indices: tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> float:
        new_values = current_vals.copy()
        indices = np.array(model_indices, dtype=int)
        new_values[indices] += allocations

        new_weights = cls._calculate_allocation_deviation(new_values, target_weights)
        return cast(float, np.sum(new_weights**2))

    @staticmethod
    def _calculate_jacobian(
        allocations: np.ndarray,
        model_indices: tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        total_value = cast(float, np.sum(current_vals)) + cast(
            float, np.sum(allocations)
        )
        indices = np.array(model_indices, dtype=int)

        active_current = current_vals[indices]
        active_targets = target_weights[indices]
        # Active allocations correspond directly to 'allocations' argument

        active_final_values = active_current + allocations

        denom: np.ndarray = active_targets * total_value
        weights: np.ndarray = (active_final_values / denom) - 1

        grad: np.ndarray = 2 * weights * (1 / denom)
        return grad

    @staticmethod
    def _calculate_hessian(
        allocations: np.ndarray,
        model_indices: tuple[int, ...],
        current_vals: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        total_value = cast(float, np.sum(current_vals)) + cast(
            float, np.sum(allocations)
        )
        indices = np.array(model_indices, dtype=int)
        active_targets = target_weights[indices]

        denom: np.ndarray = active_targets * total_value
        diag_values: np.ndarray = 2 * (1 / denom) ** 2

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

        for idx, amount_f in zip(
            best_combination, cast(list[float], best_allocations.tolist()), strict=False
        ):
            name = self.investment_names[idx]
            percent = (amount_f / additional_investment) * 100
            print(f"Optimal allocation: £{amount_f:,.2f} to '{name}' ({percent:.2f}%)")

        print()
        print(f"SSE: {optimization_result.min_sse:.2e}")
        print()

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


def _positive_float(value: str) -> float:
    """argparse type for a strictly positive float."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: '{value}'") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    """argparse type for an integer >= 1."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate optimal asset allocation.")
    _ = parser.add_argument(
        "filepath",
        help="Path to the .portfolio file",
    )
    _ = parser.add_argument(
        "additional_investment",
        type=_positive_float,
        help="Amount of additional cash to invest",
    )
    _ = parser.add_argument(
        "--n-assets",
        "-n",
        type=_positive_int,
        default=2,
        help="Number of assets to distribute investment across (default: 2)",
    )
    _ = parser.add_argument(
        "--exclude",
        "-x",
        nargs="+",
        default=[],
        help="List of asset names to exclude from optimization",
    )
    _ = parser.add_argument(
        "--taxonomy",
        "-t",
        default="Asset Allocation",
        help="Taxonomy name to use from .portfolio file (default: 'Asset Allocation')",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    filepath = cast(str, args.filepath)
    taxonomy = cast(str, args.taxonomy)
    additional_investment = cast(float, args.additional_investment)
    n_assets = cast(int, args.n_assets)
    exclude = cast(list[str], args.exclude)

    portfolio = Portfolio.from_portfolio_file(
        Path(filepath),
        taxonomy_name=taxonomy,
    )

    result = portfolio.optimize(
        additional_investment=additional_investment,
        n_assets=n_assets,
        excluded_assets=exclude,
    )
    portfolio.print_summary(result, additional_investment)


if __name__ == "__main__":
    main()
