"""Integration tests for Portfolio.optimize and Portfolio.from_portfolio_file."""

import io
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

from proto.client_pb2 import PClient
from tradey import Portfolio
from tradey.loader import _SIGNATURE
from tradey.optimizer import calculate_allocation_deviation
from tradey.optimizer import sse as calculate_sse

# Protobuf int precision constants
_SHARE_INT = int(1e8)
_PRICE_INT = int(1e8)


def _make_synthetic_portfolio_file(
    tmp_path,
    *,
    equity_weight=6000,
    bonds_weight=4000,
    global_eq_sub_weight=10000,
    global_bonds_sub_weight=10000,
    sec_a_currency="GBP",
    sec_b_currency="GBP",
    sec_a_prices=((20000, 10),),
    sec_b_prices=((20000, 5),),
    sec_a_shares=600,
    sec_b_shares=800,
):
    """Build a minimal synthetic .portfolio file with 2 securities and a taxonomy.

    Taxonomy structure (defaults):
        Asset Allocation (root)
        ├── Equity (weight=6000 → 60%)
        │   └── Global Equity (weight=100% of Equity)
        │       └── Fund A (sec-a)
        └── Bonds (weight=4000 → 40%)
            └── Global Bonds (weight=100% of Bonds)
                └── Fund B (sec-b)

    Prices are given as ``(date, close_in_pounds)`` tuples; the order in which
    they are supplied is preserved in the serialized file so callers can build
    deliberately out-of-order price series.
    """
    client = PClient()
    client.baseCurrency = "GBP"

    # Securities
    sec_a = client.securities.add()
    sec_a.uuid = "sec-a"
    sec_a.name = "Fund A"
    sec_a.currencyCode = sec_a_currency
    for date, close_pounds in sec_a_prices:
        price_a = sec_a.prices.add()
        price_a.date = date
        price_a.close = int(close_pounds * _PRICE_INT)

    sec_b = client.securities.add()
    sec_b.uuid = "sec-b"
    sec_b.name = "Fund B"
    sec_b.currencyCode = sec_b_currency
    for date, close_pounds in sec_b_prices:
        price_b = sec_b.prices.add()
        price_b.date = date
        price_b.close = int(close_pounds * _PRICE_INT)

    # Transactions
    tx_a = client.transactions.add()
    tx_a.uuid = "tx-a"
    tx_a.type = 0  # PURCHASE
    tx_a.security = "sec-a"
    tx_a.shares = sec_a_shares * _SHARE_INT

    tx_b = client.transactions.add()
    tx_b.uuid = "tx-b"
    tx_b.type = 0
    tx_b.security = "sec-b"
    tx_b.shares = sec_b_shares * _SHARE_INT

    # Taxonomy
    tax = client.taxonomies.add()
    tax.id = "tax-1"
    tax.name = "Asset Allocation"

    # Root classification (no parentId)
    root = tax.classifications.add()
    root.id = "root"
    root.name = "Asset Allocation"
    root.weight = 10000

    # Equity category
    equity = tax.classifications.add()
    equity.id = "equity"
    equity.parentId = "root"
    equity.name = "Equity"
    equity.weight = equity_weight

    # Global Equity sub-category
    global_eq = tax.classifications.add()
    global_eq.id = "global-eq"
    global_eq.parentId = "equity"
    global_eq.name = "Global Equity"
    global_eq.weight = global_eq_sub_weight
    assign_a = global_eq.assignments.add()
    assign_a.investmentVehicle = "sec-a"
    assign_a.weight = 10000

    # Bonds category
    bonds = tax.classifications.add()
    bonds.id = "bonds"
    bonds.parentId = "root"
    bonds.name = "Bonds"
    bonds.weight = bonds_weight

    # Global Bonds sub-category
    global_bonds = tax.classifications.add()
    global_bonds.id = "global-bonds"
    global_bonds.parentId = "bonds"
    global_bonds.name = "Global Bonds"
    global_bonds.weight = global_bonds_sub_weight
    assign_b = global_bonds.assignments.add()
    assign_b.investmentVehicle = "sec-b"
    assign_b.weight = 10000

    # Write to zip
    filepath = tmp_path / "synthetic.portfolio"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.portfolio", _SIGNATURE + client.SerializeToString())
    filepath.write_bytes(buf.getvalue())
    return filepath


class TestPortfolioFromFile:
    def test_loads_allocations(self, tmp_path):
        """Should parse allocations from synthetic file."""
        filepath = _make_synthetic_portfolio_file(tmp_path)
        portfolio = Portfolio.from_portfolio_file(filepath)

        assert "Global Equity" in portfolio.allocations
        assert "Global Bonds" in portfolio.allocations
        assert portfolio.allocations["Global Equity"] == pytest.approx(6000.0)
        assert portfolio.allocations["Global Bonds"] == pytest.approx(4000.0)

    def test_loads_targets(self, tmp_path):
        """Should parse allocation targets from synthetic file."""
        filepath = _make_synthetic_portfolio_file(tmp_path)
        portfolio = Portfolio.from_portfolio_file(filepath)

        # Equity 60% x Global Equity 100% = 60%
        assert portfolio.allocation_targets["Global Equity"] == pytest.approx(60.0)
        # Bonds 40% x Global Bonds 100% = 40%
        assert portfolio.allocation_targets["Global Bonds"] == pytest.approx(40.0)

    def test_missing_taxonomy_raises(self, tmp_path):
        """Should raise ValueError for unknown taxonomy name."""
        filepath = _make_synthetic_portfolio_file(tmp_path)
        with pytest.raises(ValueError, match=r"Taxonomy .* not found"):
            Portfolio.from_portfolio_file(filepath, taxonomy_name="Nonexistent")


class TestPortfolioOptimize:
    @pytest.fixture
    def balanced_portfolio(self):
        """A perfectly balanced 3-asset portfolio."""
        return Portfolio(
            allocations={"Equities": 6000.0, "Bonds": 3000.0, "Cash": 1000.0},
            allocation_targets={"Equities": 60.0, "Bonds": 30.0, "Cash": 10.0},
        )

    @pytest.fixture
    def imbalanced_portfolio(self):
        """An imbalanced portfolio where Bonds are underweight."""
        return Portfolio(
            allocations={"Equities": 7000.0, "Bonds": 2000.0, "Cash": 1000.0},
            allocation_targets={"Equities": 60.0, "Bonds": 30.0, "Cash": 10.0},
        )

    def test_result_allocations_sum_to_investment(self, balanced_portfolio):
        result = balanced_portfolio.optimize(1000.0, n_assets=2)
        assert np.sum(result.best_allocations) == pytest.approx(1000.0)

    def test_allocations_are_nonnegative(self, balanced_portfolio):
        result = balanced_portfolio.optimize(500.0, n_assets=2)
        assert all(a >= -1e-10 for a in result.best_allocations)

    def test_result_has_correct_number_of_assets(self, balanced_portfolio):
        result = balanced_portfolio.optimize(1000.0, n_assets=2)
        assert len(result.best_allocations) == 2
        assert len(result.best_combination) == 2

    def test_sse_improves_over_no_investment(self, imbalanced_portfolio):
        """Optimized allocation should have lower SSE than doing nothing."""
        current = imbalanced_portfolio.current_values
        targets = imbalanced_portfolio.target_weights

        # SSE with no additional investment
        sse_before = calculate_sse(np.array([0.0]), (0,), current, targets)

        result = imbalanced_portfolio.optimize(1000.0, n_assets=2)
        assert result.min_sse < sse_before

    def test_favours_underweight_asset(self, imbalanced_portfolio):
        """Should allocate more to Bonds (underweight) than Equities (overweight)."""
        result = imbalanced_portfolio.optimize(1000.0, n_assets=3)
        names = imbalanced_portfolio.investment_names
        alloc_map = {
            names[i]: amt
            for i, amt in zip(
                result.best_combination, result.best_allocations, strict=False
            )
        }
        assert alloc_map.get("Bonds", 0) > alloc_map.get("Equities", 0)

    def test_exclude_assets(self, imbalanced_portfolio):
        """Excluded assets should not appear in the result."""
        result = imbalanced_portfolio.optimize(
            1000.0, n_assets=2, excluded_assets=["Equities"]
        )
        names = imbalanced_portfolio.investment_names
        selected_names = [names[i] for i in result.best_combination]
        assert "Equities" not in selected_names

    def test_all_excluded_raises(self):
        portfolio = Portfolio(
            allocations={"A": 100.0},
            allocation_targets={"A": 100.0},
        )
        with pytest.raises(ValueError, match="No eligible assets"):
            portfolio.optimize(100.0, excluded_assets=["A"])

    def test_n_assets_capped_to_eligible(self, balanced_portfolio):
        """Requesting more assets than available should use all eligible."""
        result = balanced_portfolio.optimize(1000.0, n_assets=10)
        assert len(result.best_allocations) == 3

    def test_single_asset_optimization(self):
        """With n_assets=1, all investment goes to one asset."""
        portfolio = Portfolio(
            allocations={"X": 500.0, "Y": 500.0},
            allocation_targets={"X": 80.0, "Y": 20.0},
        )
        result = portfolio.optimize(200.0, n_assets=1)
        assert len(result.best_allocations) == 1
        assert result.best_allocations[0] == pytest.approx(200.0)


class _FakeMinimizeResult:
    def __init__(self, x, fun, success):
        self.x = np.asarray(x, dtype=float)
        self.fun = fun
        self.success = success


class TestOptimizeConvergence:
    """Fix #1: scipy convergence is checked; degenerate winners are rejected."""

    @pytest.fixture
    def balanced_portfolio(self):
        return Portfolio(
            allocations={"Equities": 6000.0, "Bonds": 3000.0, "Cash": 1000.0},
            allocation_targets={"Equities": 60.0, "Bonds": 30.0, "Cash": 10.0},
        )

    def test_no_combination_converges_raises(self, balanced_portfolio):
        """If every combination reports success=False, raise a clear ValueError.

        Old code ignores ``result.success`` and returns a bogus result, so this
        would NOT raise (RED).
        """

        def fake_minimize(*args, **kwargs):
            n = len(kwargs["x0"])
            return _FakeMinimizeResult([1000.0 / n] * n, 1.0, False)

        with (
            patch("tradey.optimizer.minimize", side_effect=fake_minimize),
            pytest.raises(ValueError, match="converge"),
        ):
            balanced_portfolio.optimize(1000.0, n_assets=2)

    def test_winner_failing_sum_check_raises(self, balanced_portfolio):
        """A converged winner whose allocations do not sum to the investment
        is rejected. Old code returns it silently (RED)."""

        def fake_minimize(*args, **kwargs):
            n = len(kwargs["x0"])
            return _FakeMinimizeResult([0.0] * n, 0.0, True)

        with (
            patch("tradey.optimizer.minimize", side_effect=fake_minimize),
            pytest.raises(ValueError, match="sum"),
        ):
            balanced_portfolio.optimize(1000.0, n_assets=2)

    def test_winner_with_negative_allocation_raises(self, balanced_portfolio):
        """A converged winner with a materially negative amount is rejected.
        Old code returns it silently (RED)."""

        def fake_minimize(*args, **kwargs):
            # Sums to the investment but has a large negative element.
            return _FakeMinimizeResult([1500.0, -500.0], 0.0, True)

        with (
            patch("tradey.optimizer.minimize", side_effect=fake_minimize),
            pytest.raises(ValueError, match="negative"),
        ):
            balanced_portfolio.optimize(1000.0, n_assets=2)

    def test_tiny_negative_is_clipped(self, balanced_portfolio):
        """A converged winner with a negligible negative (within -1e-9) is
        clipped to zero rather than rejected."""

        def fake_minimize(*args, **kwargs):
            return _FakeMinimizeResult([1000.0 + 1e-12, -1e-12], 0.0, True)

        with (
            patch("tradey.optimizer.minimize", side_effect=fake_minimize),
        ):
            result = balanced_portfolio.optimize(1000.0, n_assets=2)
        assert all(a >= 0.0 for a in result.best_allocations)
        assert np.sum(result.best_allocations) == pytest.approx(1000.0)


class TestZeroTargetAndEmptyPortfolio:
    """Fix #2: guard zero/absent target weights and empty portfolios."""

    def test_zero_weight_subcategory_raises(self, tmp_path):
        """A sub-category with an unset (0) target weight must raise a
        ValueError naming it. Old code produces nan and later crashes in
        scipy (RED for the friendly ValueError)."""
        filepath = _make_synthetic_portfolio_file(tmp_path, global_eq_sub_weight=0)
        with pytest.raises(ValueError, match="Global Equity"):
            Portfolio.from_portfolio_file(filepath)

    def test_empty_portfolio_raises(self, tmp_path):
        """A portfolio with no value (all holdings zero) must raise a
        ValueError. Old code yields all-nan deviations (RED)."""
        filepath = _make_synthetic_portfolio_file(
            tmp_path, sec_a_shares=0, sec_b_shares=0
        )
        with pytest.raises(ValueError, match=r"(?i)no value|empty"):
            Portfolio.from_portfolio_file(filepath)

    def test_all_zero_target_weights_raises(self, tmp_path):
        """When ALL sub-category weights are zero (targets sum to zero), the
        loader must raise the friendly target-weight ValueError, not a
        ZeroDivisionError from the normalization step (which divides by the
        zero sum before Portfolio.__post_init__ validation runs)."""
        filepath = _make_synthetic_portfolio_file(
            tmp_path, global_eq_sub_weight=0, global_bonds_sub_weight=0
        )
        with pytest.raises(ValueError, match="target weight"):
            Portfolio.from_portfolio_file(filepath)


class TestTargetNormalization:
    """Fix #3: target weights are renormalized to sum to 1.0."""

    def test_targets_below_100_are_rescaled(self, tmp_path, capsys):
        """A taxonomy summing to 80% is rescaled so a proportionally-correct
        portfolio reports ~zero deviation, and a warning is emitted.

        Old code leaves the raw 48%/32% targets, so the proportional 60/40
        portfolio shows a uniform +25% deviation (RED)."""
        # Equity 48% + Bonds 32% = 80% total; holdings 6000/4000 = 60/40.
        filepath = _make_synthetic_portfolio_file(
            tmp_path, equity_weight=4800, bonds_weight=3200
        )
        portfolio = Portfolio.from_portfolio_file(filepath)

        deviation = calculate_allocation_deviation(
            portfolio.current_values, portfolio.target_weights
        )
        np.testing.assert_allclose(deviation, 0.0, atol=1e-9)

        # Normalized targets now sum to 1.0.
        assert np.sum(portfolio.target_weights) == pytest.approx(1.0)

        err = capsys.readouterr().err
        assert "rescal" in err.lower() or "sum" in err.lower()

    def test_targets_summing_to_100_no_warning(self, tmp_path, capsys):
        """A taxonomy summing to exactly 100% is unchanged and warns nothing."""
        filepath = _make_synthetic_portfolio_file(tmp_path)  # 60/40 defaults
        portfolio = Portfolio.from_portfolio_file(filepath)

        assert portfolio.allocation_targets["Global Equity"] == pytest.approx(60.0)
        assert portfolio.allocation_targets["Global Bonds"] == pytest.approx(40.0)
        assert capsys.readouterr().err == ""


class TestCurrencyCoverage:
    """Fix #5: unresolved/blank security currencies are reported clearly."""

    def test_blank_currency_reported(self, tmp_path):
        """A security with a blank currencyCode must raise a ValueError that
        reports it distinctly. Old code raises a bare KeyError later (RED)."""
        filepath = _make_synthetic_portfolio_file(tmp_path, sec_a_currency="")
        with (
            patch("tradey.fx.fetch_exchange_rates", return_value={}),
            pytest.raises(ValueError, match="<blank>"),
        ):
            Portfolio.from_portfolio_file(filepath)

    def test_unresolved_currency_reported(self, tmp_path):
        """A currency the API does not return must raise a ValueError naming
        it. Old code raises a bare KeyError later (RED)."""
        filepath = _make_synthetic_portfolio_file(tmp_path, sec_a_currency="USD")
        with (
            patch("tradey.fx.fetch_exchange_rates", return_value={}),
            pytest.raises(ValueError, match="USD"),
        ):
            Portfolio.from_portfolio_file(filepath)


class TestLatestPriceByDate:
    """Fix #6: valuation uses the price with the maximum date, not the last."""

    def test_uses_max_date_not_last_element(self, tmp_path):
        """Prices are supplied out of date order: the last element is an older,
        higher quote. Valuation must use the latest-by-date price.

        Old code takes prices[-1] (£99) → £59,400; new code takes the max-date
        price (£10) → £6,000 (RED on old code)."""
        filepath = _make_synthetic_portfolio_file(
            tmp_path,
            sec_a_prices=((20000, 10), (10000, 99)),  # latest date first
        )
        portfolio = Portfolio.from_portfolio_file(filepath)
        assert portfolio.allocations["Global Equity"] == pytest.approx(6000.0)
