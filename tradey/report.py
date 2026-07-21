"""Report generation: pure data assembly (build_report) and rendering."""

from dataclasses import dataclass

import numpy as np
from prettytable import PrettyTable

from tradey import optimizer
from tradey.models import OptimizationResult, Portfolio

_CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}


@dataclass
class ReportRow:
    """A single investment's target and deviation before/after allocation."""

    name: str
    target: float
    old_dev: float
    new_dev: float


@dataclass
class Report:
    """Pure result of an optimization, ready to render."""

    base_currency: str
    selected_names: list[str]
    allocations: list[tuple[str, float, float]]  # (name, amount, percent)
    min_sse: float
    rows: list[ReportRow]


def build_report(
    portfolio: Portfolio,
    result: OptimizationResult,
    additional_investment: float,
) -> Report:
    """Compute report data (allocations and deviations) without any rendering."""
    best_combination = result.best_combination
    best_allocations = result.best_allocations

    selected_names = [portfolio.investment_names[i] for i in best_combination]

    allocations: list[tuple[str, float, float]] = []
    for idx, amount_f in zip(best_combination, best_allocations.tolist(), strict=False):
        name = portfolio.investment_names[idx]
        percent = (amount_f / additional_investment) * 100
        allocations.append((name, amount_f, percent))

    final_values = portfolio.current_values.copy()
    indices = np.array(best_combination, dtype=int)
    final_values[indices] += best_allocations

    targets = portfolio.target_weights
    final_dev = optimizer.calculate_allocation_deviation(final_values, targets)
    old_dev = optimizer.calculate_allocation_deviation(
        portfolio.current_values, targets
    )

    rows = [
        ReportRow(
            name=name,
            target=targets[i],
            old_dev=old_dev[i],
            new_dev=final_dev[i],
        )
        for i, name in enumerate(portfolio.investment_names)
    ]

    return Report(
        base_currency=portfolio.base_currency,
        selected_names=selected_names,
        allocations=allocations,
        min_sse=result.min_sse,
        rows=rows,
    )


def render_report(report: Report) -> str:
    """Render a Report to the console output string."""
    symbol = _CURRENCY_SYMBOLS.get(report.base_currency, f"{report.base_currency} ")

    lines: list[str] = []
    lines.append("--- Optimal Allocation Found ---")
    lines.append(f"Invest in: {report.selected_names}")
    for name, amount_f, percent in report.allocations:
        lines.append(
            f"Optimal allocation: {symbol}{amount_f:,.2f} to '{name}' ({percent:.2f}%)"
        )

    lines.append("")
    lines.append(f"SSE: {report.min_sse:.2e}")
    lines.append("")

    lines.append("--- Portfolio Weights ---")
    table = PrettyTable()
    table.field_names = ["Investment", "Target", "Old Deviance", "New Deviance"]
    table.align = "r"
    table.align["Investment"] = "l"
    for row in report.rows:
        table.add_row(
            [
                row.name,
                f"{row.target:.1%}",
                f"{row.old_dev:.1%}",
                f"{row.new_dev:.1%}",
            ]
        )
    lines.append(str(table))

    return "\n".join(lines)
