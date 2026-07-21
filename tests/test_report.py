"""Golden-output tests for report rendering (audit HIGH).

``render_report`` is a pure ``Report -> str`` function, so a fixed portfolio and
optimization result must produce a byte-for-byte fixed string. The two variants
lock the currency-symbol behaviour (the Task 3 fix): GBP renders ``£`` and USD
renders ``$`` on the allocation line, with everything else identical.
"""

import numpy as np

from tradey import report
from tradey.models import OptimizationResult, Portfolio

# Fixed inputs: a 70/30 held portfolio against a 60/40 target, investing 1000
# entirely into Bonds. Deviances are hand-checkable:
#   old total 10000: Equity 7000/(0.6*10000)-1 = +16.7%, Bonds 3000/4000-1 = -25.0%
#   new total 11000: Equity 7000/6600-1 = +6.1%,  Bonds 4000/4400-1 = -9.1%
_GBP_GOLDEN = """\
--- Optimal Allocation Found ---
Invest in: ['Bonds']
Optimal allocation: £1,000.00 to 'Bonds' (100.00%)

SSE: 1.19e-02

--- Portfolio Weights ---
+------------+--------+--------------+--------------+
| Investment | Target | Old Deviance | New Deviance |
+------------+--------+--------------+--------------+
| Equity     |  60.0% |        16.7% |         6.1% |
| Bonds      |  40.0% |       -25.0% |        -9.1% |
+------------+--------+--------------+--------------+"""

# Identical to the GBP golden except for the currency symbol on the allocation
# line ($ instead of £).
_USD_GOLDEN = """\
--- Optimal Allocation Found ---
Invest in: ['Bonds']
Optimal allocation: $1,000.00 to 'Bonds' (100.00%)

SSE: 1.19e-02

--- Portfolio Weights ---
+------------+--------+--------------+--------------+
| Investment | Target | Old Deviance | New Deviance |
+------------+--------+--------------+--------------+
| Equity     |  60.0% |        16.7% |         6.1% |
| Bonds      |  40.0% |       -25.0% |        -9.1% |
+------------+--------+--------------+--------------+"""


def _build_fixture_report(base_currency: str) -> report.Report:
    portfolio = Portfolio(
        allocations={"Equity": 7000.0, "Bonds": 3000.0},
        allocation_targets={"Equity": 60.0, "Bonds": 40.0},
        base_currency=base_currency,
    )
    result = OptimizationResult(
        best_combination=(1,),
        best_allocations=np.array([1000.0]),
        min_sse=0.0119,
    )
    return report.build_report(portfolio, result, additional_investment=1000.0)


class TestRenderReportGolden:
    def test_gbp_currency_symbol(self):
        rendered = report.render_report(_build_fixture_report("GBP"))
        assert rendered == _GBP_GOLDEN

    def test_usd_currency_symbol(self):
        rendered = report.render_report(_build_fixture_report("USD"))
        assert rendered == _USD_GOLDEN
