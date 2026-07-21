"""Integration tests for Portfolio.optimize and Portfolio.from_portfolio_file.

The synthetic ``.portfolio`` builder lives in ``conftest.py`` (exposed as the
``make_portfolio_file`` / ``make_portfolio_zip`` fixtures).
"""

from unittest.mock import patch

import numpy as np
import pytest

from proto.client_pb2 import PClient
from tradey import Portfolio
from tradey.loader import load_portfolio
from tradey.optimizer import calculate_allocation_deviation
from tradey.optimizer import sse as calculate_sse


class TestPortfolioFromFile:
    def test_loads_allocations(self, make_portfolio_file):
        """Should parse allocations from synthetic file."""
        filepath = make_portfolio_file()
        portfolio = Portfolio.from_portfolio_file(filepath)

        assert "Global Equity" in portfolio.allocations
        assert "Global Bonds" in portfolio.allocations
        assert portfolio.allocations["Global Equity"] == pytest.approx(6000.0)
        assert portfolio.allocations["Global Bonds"] == pytest.approx(4000.0)

    def test_loads_targets(self, make_portfolio_file):
        """Should parse allocation targets from synthetic file."""
        filepath = make_portfolio_file()
        portfolio = Portfolio.from_portfolio_file(filepath)

        # Equity 60% x Global Equity 100% = 60%
        assert portfolio.allocation_targets["Global Equity"] == pytest.approx(60.0)
        # Bonds 40% x Global Bonds 100% = 40%
        assert portfolio.allocation_targets["Global Bonds"] == pytest.approx(40.0)

    def test_missing_taxonomy_raises(self, make_portfolio_file):
        """Should raise ValueError for unknown taxonomy name."""
        filepath = make_portfolio_file()
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


class TestOptimizeEdgeCases:
    """Boundary behaviour of ``optimize`` around n_assets and portfolio size.

    (audit MEDIUM: optimizer edge cases.)
    """

    @pytest.fixture
    def three_asset_portfolio(self):
        return Portfolio(
            allocations={"A": 6000.0, "B": 3000.0, "C": 1000.0},
            allocation_targets={"A": 60.0, "B": 30.0, "C": 10.0},
        )

    def test_n_assets_equals_portfolio_size(self, three_asset_portfolio):
        """n_assets exactly equal to the number of holdings splits across all."""
        result = three_asset_portfolio.optimize(900.0, n_assets=3)
        assert len(result.best_allocations) == 3
        assert len(result.best_combination) == 3
        assert np.sum(result.best_allocations) == pytest.approx(900.0)

    def test_n_assets_exceeds_eligible_after_exclusion(self, three_asset_portfolio):
        """Excluding one holding leaves 2 eligible; asking for 3 degrades to 2.

        Pins the graceful cap in ``optimizer.optimize`` (n_assets is clamped to
        the eligible count rather than producing empty combinations or raising).
        """
        result = three_asset_portfolio.optimize(
            1000.0, n_assets=3, excluded_assets=["A"]
        )
        assert len(result.best_allocations) == 2
        names = three_asset_portfolio.investment_names
        selected = [names[i] for i in result.best_combination]
        assert "A" not in selected
        assert np.sum(result.best_allocations) == pytest.approx(1000.0)

    def test_single_asset_portfolio(self):
        """A portfolio with a single holding: n_assets caps to 1, all cash to it."""
        portfolio = Portfolio(
            allocations={"Only": 5000.0},
            allocation_targets={"Only": 100.0},
        )
        result = portfolio.optimize(500.0, n_assets=2)
        assert result.best_combination == (0,)
        assert len(result.best_allocations) == 1
        assert result.best_allocations[0] == pytest.approx(500.0)


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

    def test_zero_weight_subcategory_raises(self, make_portfolio_file):
        """A sub-category with an unset (0) target weight must raise a
        ValueError naming it. Old code produces nan and later crashes in
        scipy (RED for the friendly ValueError)."""
        filepath = make_portfolio_file(global_eq_sub_weight=0)
        with pytest.raises(ValueError, match="Global Equity"):
            Portfolio.from_portfolio_file(filepath)

    def test_empty_portfolio_raises(self, make_portfolio_file):
        """A portfolio with no value (all holdings zero) must raise a
        ValueError. Old code yields all-nan deviations (RED)."""
        filepath = make_portfolio_file(sec_a_shares=0, sec_b_shares=0)
        with pytest.raises(ValueError, match=r"(?i)no value|empty"):
            Portfolio.from_portfolio_file(filepath)

    def test_all_zero_target_weights_raises(self, make_portfolio_file):
        """When ALL sub-category weights are zero (targets sum to zero), the
        loader must raise the friendly target-weight ValueError, not a
        ZeroDivisionError from the normalization step (which divides by the
        zero sum before Portfolio.__post_init__ validation runs)."""
        filepath = make_portfolio_file(
            global_eq_sub_weight=0, global_bonds_sub_weight=0
        )
        with pytest.raises(ValueError, match="target weight"):
            Portfolio.from_portfolio_file(filepath)


class TestTargetNormalization:
    """Fix #3: target weights are renormalized to sum to 1.0."""

    def test_targets_below_100_are_rescaled(self, make_portfolio_file, capsys):
        """A taxonomy summing to 80% is rescaled so a proportionally-correct
        portfolio reports ~zero deviation, and a warning is emitted.

        Old code leaves the raw 48%/32% targets, so the proportional 60/40
        portfolio shows a uniform +25% deviation (RED)."""
        # Equity 48% + Bonds 32% = 80% total; holdings 6000/4000 = 60/40.
        filepath = make_portfolio_file(equity_weight=4800, bonds_weight=3200)
        portfolio = Portfolio.from_portfolio_file(filepath)

        deviation = calculate_allocation_deviation(
            portfolio.current_values, portfolio.target_weights
        )
        np.testing.assert_allclose(deviation, 0.0, atol=1e-9)

        # Normalized targets now sum to 1.0.
        assert np.sum(portfolio.target_weights) == pytest.approx(1.0)

        err = capsys.readouterr().err
        assert "rescal" in err.lower() or "sum" in err.lower()

    def test_targets_summing_to_100_no_warning(self, make_portfolio_file, capsys):
        """A taxonomy summing to exactly 100% is unchanged and warns nothing."""
        filepath = make_portfolio_file()  # 60/40 defaults
        portfolio = Portfolio.from_portfolio_file(filepath)

        assert portfolio.allocation_targets["Global Equity"] == pytest.approx(60.0)
        assert portfolio.allocation_targets["Global Bonds"] == pytest.approx(40.0)
        assert capsys.readouterr().err == ""


class TestCurrencyCoverage:
    """Fix #5: unresolved/blank security currencies are reported clearly."""

    def test_blank_currency_reported(self, make_portfolio_file):
        """A security with a blank currencyCode must raise a ValueError that
        reports it distinctly. Old code raises a bare KeyError later (RED)."""
        filepath = make_portfolio_file(sec_a_currency="")
        with (
            patch("tradey.fx.fetch_exchange_rates", return_value={}),
            pytest.raises(ValueError, match="<blank>"),
        ):
            Portfolio.from_portfolio_file(filepath)

    def test_unresolved_currency_reported(self, make_portfolio_file):
        """A currency the API does not return must raise a ValueError naming
        it. Old code raises a bare KeyError later (RED)."""
        filepath = make_portfolio_file(sec_a_currency="USD")
        with (
            patch("tradey.fx.fetch_exchange_rates", return_value={}),
            pytest.raises(ValueError, match="USD"),
        ):
            Portfolio.from_portfolio_file(filepath)


class TestCurrencyConversionValue:
    """The FX conversion factor is applied to security values end-to-end.

    (audit HIGH: currency-conversion arithmetic was never exercised through the
    loader — only the missing/blank-code error paths were.)
    """

    def test_foreign_security_value_uses_factor(self, make_portfolio_file):
        """A USD-priced security's value is ``shares x price x factor``.

        Designed to fail if the factor were inverted: with 600 shares @ $10 and
        a USD factor of 0.5 the correct value is 3000, whereas an inverted
        factor (2.0) would give 12000.
        """
        filepath = make_portfolio_file(sec_a_currency="USD")

        def fake_rates(base_currency, currencies):
            assert base_currency == "GBP"
            assert "USD" in currencies
            return {"GBP": 1.0, "USD": 0.5}

        portfolio = load_portfolio(filepath, fetch_rates=fake_rates)

        # sec-a: 600 shares x $10 x 0.5 = 3000.0
        assert portfolio.allocations["Global Equity"] == pytest.approx(3000.0)
        # sec-b (GBP) is unaffected: 800 shares x £5 x 1.0 = 4000.0
        assert portfolio.allocations["Global Bonds"] == pytest.approx(4000.0)


class TestLoaderBranches:
    """Loader branch coverage: no-root, baseCurrency fallback, taxonomy select.

    (audit MEDIUM: from_portfolio_file error/branch paths.)
    """

    def test_no_root_classification_raises(self, make_portfolio_zip):
        """A taxonomy where every classification has a parentId has no root."""
        client = PClient()
        client.baseCurrency = "GBP"
        tax = client.taxonomies.add()
        tax.id = "tax-1"
        tax.name = "Asset Allocation"
        # Single classification pointing at a non-existent parent → no root.
        orphan = tax.classifications.add()
        orphan.id = "orphan"
        orphan.parentId = "does-not-exist"
        orphan.name = "Orphan"
        orphan.weight = 10000

        filepath = make_portfolio_zip(client.SerializeToString())
        with pytest.raises(ValueError, match="no root"):
            load_portfolio(filepath, fetch_rates=lambda base, currencies: {base: 1.0})

    def test_base_currency_defaults_to_gbp(self, make_portfolio_file):
        """An unset client.baseCurrency falls back to GBP."""
        filepath = make_portfolio_file(base_currency=None)
        portfolio = load_portfolio(
            filepath, fetch_rates=lambda base, currencies: {"GBP": 1.0}
        )
        assert portfolio.base_currency == "GBP"

    def test_taxonomy_name_selects_correct_taxonomy(self, make_portfolio_file):
        """Passing taxonomy_name picks the matching taxonomy, not the first."""
        filepath = make_portfolio_file(second_taxonomy_name="Region")

        default = load_portfolio(filepath)  # default "Asset Allocation"
        assert "Global Equity" in default.allocations
        assert "Alt Sub" not in default.allocations

        region = load_portfolio(filepath, taxonomy_name="Region")
        assert "Alt Sub" in region.allocations
        assert "Global Equity" not in region.allocations


class TestLatestPriceByDate:
    """Fix #6: valuation uses the price with the maximum date, not the last."""

    def test_uses_max_date_not_last_element(self, make_portfolio_file):
        """Prices are supplied out of date order: the last element is an older,
        higher quote. Valuation must use the latest-by-date price.

        Old code takes prices[-1] (£99) → £59,400; new code takes the max-date
        price (£10) → £6,000 (RED on old code)."""
        filepath = make_portfolio_file(
            sec_a_prices=((20000, 10), (10000, 99)),  # latest date first
        )
        portfolio = Portfolio.from_portfolio_file(filepath)
        assert portfolio.allocations["Global Equity"] == pytest.approx(6000.0)
