"""Command-line interface: argument parsing and wiring."""

import argparse
from pathlib import Path
from typing import NamedTuple

from tradey import report
from tradey.loader import load_portfolio


class Args(NamedTuple):
    """Typed view of the parsed command-line arguments."""

    filepath: str
    additional_investment: float
    n_assets: int
    exclude: list[str]
    taxonomy: str


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
    ns = parse_arguments()
    args = Args(
        filepath=ns.filepath,
        additional_investment=ns.additional_investment,
        n_assets=ns.n_assets,
        exclude=ns.exclude,
        taxonomy=ns.taxonomy,
    )

    portfolio = load_portfolio(
        Path(args.filepath),
        taxonomy_name=args.taxonomy,
    )

    result = portfolio.optimize(
        additional_investment=args.additional_investment,
        n_assets=args.n_assets,
        excluded_assets=args.exclude,
    )
    output = report.build_report(portfolio, result, args.additional_investment)
    print(report.render_report(output))


if __name__ == "__main__":
    main()
