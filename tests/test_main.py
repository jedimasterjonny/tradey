"""End-to-end test for cli.main (audit HIGH).

This is the only test that wires argv -> load_portfolio -> optimize ->
build_report -> render_report -> print together; every other test exercises
those stages in isolation.
"""

import json
import sys
from unittest.mock import patch

from tradey import cli


class _FakeResponse:
    """Minimal context-manager stand-in for an HTTP response."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


class TestMainEndToEnd:
    def test_all_gbp_runs_and_prints_report(self, make_portfolio_file, capsys):
        """A GBP-only portfolio needs no FX; main() prints the full report."""
        filepath = make_portfolio_file()
        argv = ["tradey", str(filepath), "1000"]

        with patch.object(sys, "argv", argv):
            cli.main()  # must return without raising / exiting

        out = capsys.readouterr().out
        # The two section headers, the allocation line and the deviance table.
        assert "--- Optimal Allocation Found ---" in out
        assert "Optimal allocation:" in out
        assert "--- Portfolio Weights ---" in out
        assert "Global Equity" in out
        assert "Global Bonds" in out
        # base currency is GBP → pound symbol on the allocation line.
        assert "£" in out

    def test_foreign_currency_stubs_network_at_fx_boundary(
        self, make_portfolio_file, capsys
    ):
        """A USD-priced security drives a real resolve_currency_factors call.

        main() offers no fetch injection, so the network is stubbed at the fx
        boundary (``urllib.request.urlopen``). Rate 1.25 → factor 0.8.
        """
        filepath = make_portfolio_file(sec_a_currency="USD")
        payload = json.dumps({"rates": {"USD": 1.25}}).encode()
        argv = ["tradey", str(filepath), "1000"]

        with (
            patch.object(sys, "argv", argv),
            patch(
                "tradey.fx.urllib.request.urlopen",
                return_value=_FakeResponse(payload),
            ),
        ):
            cli.main()

        out = capsys.readouterr().out
        assert "--- Optimal Allocation Found ---" in out
        assert "Optimal allocation:" in out
        assert "--- Portfolio Weights ---" in out
        assert "Global Equity" in out
