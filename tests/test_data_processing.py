"""Tests for _compute_shares, _resolve_currency_factors, and _read_client."""

import io
import zipfile
from unittest.mock import patch

import pytest

from proto.client_pb2 import PClient, PTransaction
from tradey import (
    _SHARE_PRECISION,
    _SIGNATURE,
    _compute_shares,
    _read_client,
    _resolve_currency_factors,
)

SHARES = int(_SHARE_PRECISION)


class TestComputeShares:
    def _make_transaction(self, security: str, tx_type: int, shares: int):
        """Build a synthetic PTransaction."""
        t = PTransaction()
        t.security = security
        t.type = tx_type
        t.shares = shares
        return t

    def test_single_purchase(self):
        client = PClient()
        client.transactions.append(self._make_transaction("sec-1", 0, 100 * SHARES))
        result = _compute_shares(client)
        assert result["sec-1"] == 100 * SHARES

    def test_purchase_then_sale(self):
        client = PClient()
        client.transactions.append(self._make_transaction("sec-1", 0, 100 * SHARES))
        client.transactions.append(self._make_transaction("sec-1", 1, 40 * SHARES))
        result = _compute_shares(client)
        assert result["sec-1"] == 60 * SHARES

    def test_inbound_and_outbound_delivery(self):
        """Types 2 (inbound) and 3 (outbound) should add/subtract."""
        client = PClient()
        client.transactions.append(self._make_transaction("sec-2", 2, 200 * SHARES))
        client.transactions.append(self._make_transaction("sec-2", 3, 50 * SHARES))
        result = _compute_shares(client)
        assert result["sec-2"] == 150 * SHARES

    def test_multiple_securities(self):
        client = PClient()
        client.transactions.append(self._make_transaction("sec-a", 0, 10 * SHARES))
        client.transactions.append(self._make_transaction("sec-b", 0, 20 * SHARES))
        result = _compute_shares(client)
        assert result["sec-a"] == 10 * SHARES
        assert result["sec-b"] == 20 * SHARES

    def test_transaction_without_security_is_skipped(self):
        """Transactions with no security field (e.g. deposits) should be ignored."""
        client = PClient()
        t = PTransaction()
        t.type = 6  # DEPOSIT
        t.amount = 100000
        client.transactions.append(t)
        result = _compute_shares(client)
        assert result == {}

    def test_irrelevant_type_ignored(self):
        """Transaction types outside 0-3 should not modify shares."""
        client = PClient()
        # Type 8 = DIVIDEND — has a security but shouldn't change shares
        client.transactions.append(self._make_transaction("sec-1", 8, 0))
        result = _compute_shares(client)
        assert result == {}


class TestResolveCurrencyFactors:
    def test_base_currency_always_one(self):
        factors = _resolve_currency_factors("GBP", set())
        assert factors["GBP"] == 1.0

    def test_fixed_gbx_to_gbp(self):
        factors = _resolve_currency_factors("GBP", {"GBX"})
        assert factors["GBX"] == pytest.approx(0.01)
        assert factors["GBP"] == 1.0

    def test_fetches_unknown_currency(self):
        """Unknown currencies should trigger an API fetch."""
        mock_rates = {"USD": 0.8}
        with patch("tradey._fetch_exchange_rates", return_value=mock_rates) as mock:
            factors = _resolve_currency_factors("GBP", {"USD"})
            mock.assert_called_once_with("GBP", {"USD"})
            assert factors["USD"] == pytest.approx(0.8)

    def test_mixed_fixed_and_fetched(self):
        """GBX should use fixed rate, EUR should be fetched."""
        mock_rates = {"EUR": 1.15}
        with patch("tradey._fetch_exchange_rates", return_value=mock_rates) as mock:
            factors = _resolve_currency_factors("GBP", {"GBX", "EUR"})
            # Only EUR should be fetched, not GBX
            mock.assert_called_once_with("GBP", {"EUR"})
            assert factors["GBX"] == pytest.approx(0.01)
            assert factors["EUR"] == pytest.approx(1.15)


class TestReadClient:
    def _make_portfolio_file(self, tmp_path, payload: bytes, signature=_SIGNATURE):
        """Create a synthetic .portfolio zip file."""
        filepath = tmp_path / "test.portfolio"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.portfolio", signature + payload)
        filepath.write_bytes(buf.getvalue())
        return filepath

    def test_reads_valid_file(self, tmp_path):
        """Should parse a valid .portfolio file with minimal protobuf data."""
        client = PClient()
        client.version = 1
        client.baseCurrency = "GBP"
        sec = client.securities.add()
        sec.uuid = "test-uuid"
        sec.name = "Test Fund"
        sec.currencyCode = "GBP"

        filepath = self._make_portfolio_file(tmp_path, client.SerializeToString())
        result = _read_client(filepath)

        assert result.version == 1
        assert result.baseCurrency == "GBP"
        assert len(result.securities) == 1
        assert result.securities[0].name == "Test Fund"

    def test_invalid_signature_raises(self, tmp_path):
        """Should raise ValueError for files with wrong signature."""
        filepath = self._make_portfolio_file(tmp_path, b"somedata", signature=b"BADsig")
        with pytest.raises(ValueError, match="Not a Portfolio Performance protobuf"):
            _read_client(filepath)

    def test_roundtrip_with_transactions(self, tmp_path):
        """Transactions should survive the serialization roundtrip."""
        client = PClient()
        t = client.transactions.add()
        t.uuid = "tx-1"
        t.type = 0  # PURCHASE
        t.security = "sec-1"
        t.shares = 50 * SHARES

        filepath = self._make_portfolio_file(tmp_path, client.SerializeToString())
        result = _read_client(filepath)

        assert len(result.transactions) == 1
        assert result.transactions[0].shares == 50 * SHARES

    def test_non_zip_file_raises(self, tmp_path):
        """A file that isn't a zip should raise a friendly ValueError."""
        filepath = tmp_path / "test.portfolio"
        filepath.write_bytes(b"this is not a zip file")
        with pytest.raises(ValueError, match="not a valid Portfolio Performance file"):
            _read_client(filepath)

    def test_missing_data_member_raises(self, tmp_path):
        """A zip missing the 'data.portfolio' member should raise ValueError."""
        filepath = tmp_path / "test.portfolio"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", b"nothing useful here")
        filepath.write_bytes(buf.getvalue())
        with pytest.raises(ValueError, match="not a valid Portfolio Performance file"):
            _read_client(filepath)

    def test_corrupt_protobuf_raises(self, tmp_path):
        """A valid signature with a corrupt protobuf should raise ValueError."""
        filepath = self._make_portfolio_file(tmp_path, b"\x08")
        with pytest.raises(ValueError, match="not a valid Portfolio Performance file"):
            _read_client(filepath)
