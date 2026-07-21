"""Shared fixtures, helpers and constants for the tradey test suite.

Centralises the ``.portfolio`` zip/protobuf builders so the signature and
precision logic lives in exactly one place, removing the duplicated helpers,
duplicated precision constants and magic date value that were previously
scattered across the individual test modules.
"""

import io
import zipfile

import pytest

from proto.client_pb2 import PClient

# Precision/signature constants are imported from the loader rather than
# redefined, so fixtures cannot silently diverge from production if the scaling
# ever changes.
from tradey.loader import _PRICE_PRECISION, _SHARE_PRECISION, _SIGNATURE

# Protobuf stores share counts and prices as integers scaled by these factors.
_SHARE_INT = int(_SHARE_PRECISION)
_PRICE_INT = int(_PRICE_PRECISION)

# Portfolio Performance records a price's ``date`` as a plain integer (see
# proto/client_pb2.pyi: ``date: int``). The repo ships no .proto source or field
# documentation, so the exact epoch/units are not verifiable here; this is
# simply an arbitrary valid date int. Only the *relative* ordering of dates is
# load-bearing for valuation (``security_value_base`` picks max-by-date), so any
# distinct positive ints suffice for tests.
_PRICE_DATE = 20000


def write_portfolio_zip(
    tmp_path,
    payload: bytes,
    *,
    signature: bytes = _SIGNATURE,
    filename: str = "test.portfolio",
):
    """Write ``signature + payload`` as the ``data.portfolio`` member of a zip.

    Low-level helper shared by the synthetic-portfolio builder and the
    ``load_client`` signature/corruption tests.
    """
    filepath = tmp_path / filename
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.portfolio", signature + payload)
    filepath.write_bytes(buf.getvalue())
    return filepath


def build_synthetic_portfolio_file(
    tmp_path,
    *,
    equity_weight=6000,
    bonds_weight=4000,
    global_eq_sub_weight=10000,
    global_bonds_sub_weight=10000,
    sec_a_currency="GBP",
    sec_b_currency="GBP",
    sec_a_prices=((_PRICE_DATE, 10),),
    sec_b_prices=((_PRICE_DATE, 5),),
    sec_a_shares=600,
    sec_b_shares=800,
    base_currency="GBP",
    second_taxonomy_name=None,
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

    Args:
        base_currency: value for ``client.baseCurrency``; pass ``None`` to leave
            it unset (exercises the loader's GBP fallback).
        second_taxonomy_name: when given, a second, differently-named taxonomy
            (sub-category "Alt Sub" holding sec-a) is appended so taxonomy
            selection can be tested.
    """
    client = PClient()
    if base_currency is not None:
        client.baseCurrency = base_currency

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

    # Optional second taxonomy for selection tests. Distinct sub-category name
    # ("Alt Sub") so a test can prove which taxonomy was chosen.
    if second_taxonomy_name is not None:
        tax2 = client.taxonomies.add()
        tax2.id = "tax-2"
        tax2.name = second_taxonomy_name
        root2 = tax2.classifications.add()
        root2.id = "root2"
        root2.name = second_taxonomy_name
        root2.weight = 10000
        alt = tax2.classifications.add()
        alt.id = "alt"
        alt.parentId = "root2"
        alt.name = "Alt"
        alt.weight = 10000
        alt_sub = tax2.classifications.add()
        alt_sub.id = "alt-sub"
        alt_sub.parentId = "alt"
        alt_sub.name = "Alt Sub"
        alt_sub.weight = 10000
        alt_assign = alt_sub.assignments.add()
        alt_assign.investmentVehicle = "sec-a"
        alt_assign.weight = 10000

    return write_portfolio_zip(
        tmp_path,
        client.SerializeToString(),
        filename="synthetic.portfolio",
    )


@pytest.fixture
def make_portfolio_file(tmp_path):
    """Return a factory that builds a synthetic .portfolio file in ``tmp_path``.

    Usage: ``path = make_portfolio_file(equity_weight=4800, ...)``.
    """

    def _make(**kwargs):
        return build_synthetic_portfolio_file(tmp_path, **kwargs)

    return _make


@pytest.fixture
def make_portfolio_zip(tmp_path):
    """Return a factory that writes a raw ``signature + payload`` zip in ``tmp_path``.

    Usage: ``path = make_portfolio_zip(payload_bytes, signature=b"BADsig")``.
    """

    def _make(payload: bytes, *, signature: bytes = _SIGNATURE):
        return write_portfolio_zip(tmp_path, payload, signature=signature)

    return _make
