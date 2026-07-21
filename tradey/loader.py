"""Load a Portfolio Performance ``.portfolio`` export into a Portfolio.

Handles file/zip/protobuf I/O, share computation, taxonomy-tree walking and
valuation. Network FX is injected (see ``load_portfolio``) so this module is
testable offline.
"""

import sys
import zipfile
from enum import IntEnum
from pathlib import Path

from google.protobuf.message import DecodeError

from proto.client_pb2 import PClient
from tradey.fx import resolve_currency_factors
from tradey.models import Portfolio

# Precision constants used by Portfolio Performance
_SHARE_PRECISION = 1e8
_PRICE_PRECISION = 1e8

_SIGNATURE = b"PPPBV1"

# Portfolio Performance stores a category's weight in 1/10000ths of the total.
_CATEGORY_WEIGHT_SCALE = 10_000.0
# A sub-category's weight is a percentage (0-100) of its parent category.
_SUB_WEIGHT_SCALE = 100.0


class TxType(IntEnum):
    """Transaction types.

    Values mirror the transaction ``type`` enum in the Portfolio Performance
    client protobuf; the numeric codes come straight from that schema.
    """

    PURCHASE = 0
    SALE = 1
    INBOUND_DELIVERY = 2
    OUTBOUND_DELIVERY = 3


def load_client(filepath: Path) -> PClient:
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


def compute_shares(client: PClient) -> dict[str, int]:
    """Calculate shares held per security from the transaction history."""
    sec_shares: dict[str, int] = {}
    for t in client.transactions:
        if not t.HasField("security"):
            continue
        sid = t.security
        shares = t.shares if t.HasField("shares") else 0
        if t.type in (TxType.PURCHASE, TxType.INBOUND_DELIVERY):
            sec_shares[sid] = sec_shares.get(sid, 0) + shares
        elif t.type in (TxType.SALE, TxType.OUTBOUND_DELIVERY):
            sec_shares[sid] = sec_shares.get(sid, 0) - shares
    return sec_shares


def security_value_base(sec, shares_held: int, factors: dict[str, float]) -> float:
    """Calculate the current value of a security in base currency."""
    if sec is None or not sec.prices:
        return 0.0
    shares = shares_held / _SHARE_PRECISION
    latest_price = max(sec.prices, key=lambda p: p.date)
    price = latest_price.close / _PRICE_PRECISION
    return shares * price * factors[sec.currencyCode]


def build_allocations(
    taxonomy,
    sec_map: dict,
    sec_shares: dict[str, int],
    factors: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Walk the taxonomy tree, returning (allocations, allocation_targets).

    Allocations are current values in base currency, keyed by sub-category name.
    Allocation targets are percentages (0-100) of the whole portfolio.
    """
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

    allocations: dict[str, float] = {}
    allocation_targets: dict[str, float] = {}

    for cat_id in top_level_ids:
        cat = classifications[cat_id]
        cat_weight = cat.weight / _CATEGORY_WEIGHT_SCALE  # as fraction
        sub_ids = children.get(cat_id, [])

        for sub_id in sub_ids:
            sub = classifications[sub_id]

            # Sum values of all securities assigned to this sub-category
            sub_value = sum(
                security_value_base(
                    sec_map.get(a.investmentVehicle),
                    sec_shares.get(a.investmentVehicle, 0),
                    factors,
                )
                for a in sub.assignments
            )
            allocations[sub.name] = sub_value

            # Target = sub_weight * category_weight (both as fractions of parent)
            sub_weight_pct = sub.weight / _SUB_WEIGHT_SCALE
            allocation_targets[sub.name] = sub_weight_pct * cat_weight

    return allocations, allocation_targets


def load_portfolio(
    filepath,
    taxonomy_name: str = "Asset Allocation",
    fetch_rates=resolve_currency_factors,
) -> Portfolio:
    """Parse a .portfolio file and return a Portfolio instance.

    Args:
        filepath: Path to the .portfolio file (zip containing protobuf data).
        taxonomy_name: Name of the taxonomy to use for allocation targets.
        fetch_rates: Callable ``(base_currency, currencies) -> factors`` used to
            resolve currency conversion factors. Injectable so tests can stub FX.
    """
    client = load_client(Path(filepath))

    base_currency: str = client.baseCurrency or "GBP"
    sec_map = {s.uuid: s for s in client.securities}
    sec_shares = compute_shares(client)

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

    # Resolve currency conversion factors
    assigned_currencies: set[str] = set()
    for c in taxonomy.classifications:
        for a in c.assignments:
            sec = sec_map.get(a.investmentVehicle)
            if sec:
                assigned_currencies.add(sec.currencyCode)
    currency_factors = fetch_rates(base_currency, assigned_currencies)

    # Every assigned currency must have a resolved conversion factor,
    # otherwise valuation would later raise a bare KeyError. A blank
    # currencyCode is reported distinctly.
    missing_currencies = sorted(
        (c if c else "<blank>")
        for c in assigned_currencies
        if c not in currency_factors
    )
    if missing_currencies:
        raise ValueError(
            "securities priced in currencies with no exchange rate: "
            f"{missing_currencies}"
        )

    allocations, allocation_targets = build_allocations(
        taxonomy, sec_map, sec_shares, currency_factors
    )

    # Normalize target weights so the derived fractions sum to 1.0. Real
    # taxonomies often do not sum to 100% (unclassified cash, partial weights),
    # which would bias every deviation. allocation_targets are stored as
    # percentages, so after rescaling the dict still sums to 100 while the
    # Portfolio's derived target_weights (dict / 100) sum to 1.0.
    raw_sum = sum(allocation_targets.values()) / 100.0  # as a fraction
    # Guard the division below: if no sub-category carries a positive target
    # (raw_sum <= 0), normalization would divide by zero before the Portfolio
    # target>0 validation runs. Raise the same friendly error here instead.
    if raw_sum <= 0:
        zero_target_names = [
            name for name, target in allocation_targets.items() if target <= 0
        ]
        raise ValueError(
            "the following sub-categories have no target weight (weight must "
            f"be greater than 0): {zero_target_names}"
        )
    if abs(raw_sum - 1.0) > 0.01:
        print(
            f"warning: taxonomy target weights sum to {raw_sum:.2%} "
            "(expected 100%); rescaling targets to sum to 1.0",
            file=sys.stderr,
        )
    for name in allocation_targets:
        allocation_targets[name] /= raw_sum

    return Portfolio(
        allocations=allocations,
        allocation_targets=allocation_targets,
        base_currency=base_currency,
    )
