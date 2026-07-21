"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest

from tradey import parse_arguments


class TestParseArguments:
    def _parse(self, args: list[str]):
        """Call parse_arguments with the given CLI args."""
        with patch.object(sys, "argv", ["tradey", *args]):
            return parse_arguments()

    def test_required_args(self):
        args = self._parse(["portfolio.portfolio", "1000"])
        assert args.filepath == "portfolio.portfolio"
        assert args.additional_investment == 1000.0

    def test_default_n_assets(self):
        args = self._parse(["f.portfolio", "500"])
        assert args.n_assets == 2

    def test_custom_n_assets_long(self):
        args = self._parse(["f.portfolio", "500", "--n-assets", "5"])
        assert args.n_assets == 5

    def test_custom_n_assets_short(self):
        args = self._parse(["f.portfolio", "500", "-n", "3"])
        assert args.n_assets == 3

    def test_default_exclude(self):
        args = self._parse(["f.portfolio", "500"])
        assert args.exclude == []

    def test_exclude_single(self):
        args = self._parse(["f.portfolio", "500", "--exclude", "Bonds"])
        assert args.exclude == ["Bonds"]

    def test_exclude_multiple(self):
        args = self._parse(["f.portfolio", "500", "-x", "Bonds", "Cash", "Property"])
        assert args.exclude == ["Bonds", "Cash", "Property"]

    def test_default_taxonomy(self):
        args = self._parse(["f.portfolio", "500"])
        assert args.taxonomy == "Asset Allocation"

    def test_custom_taxonomy_long(self):
        args = self._parse(["f.portfolio", "500", "--taxonomy", "Region"])
        assert args.taxonomy == "Region"

    def test_custom_taxonomy_short(self):
        args = self._parse(["f.portfolio", "500", "-t", "Sector"])
        assert args.taxonomy == "Sector"

    def test_investment_is_float(self):
        args = self._parse(["f.portfolio", "1234.56"])
        assert args.additional_investment == pytest.approx(1234.56)

    def test_missing_required_args_exits(self):
        with pytest.raises(SystemExit):
            self._parse([])

    def test_missing_investment_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["f.portfolio"])

    def test_zero_investment_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["f.portfolio", "0"])

    def test_negative_investment_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["f.portfolio", "-100"])

    def test_zero_n_assets_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["f.portfolio", "1000", "-n", "0"])

    def test_negative_n_assets_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["f.portfolio", "1000", "-n", "-1"])
