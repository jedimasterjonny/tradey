"""Currency conversion: fixed factors plus live exchange-rate fetching.

All network access lives here so the loader and optimizer stay testable offline.
"""

import json
import math
import urllib.request
from urllib.error import URLError

# Fixed currency conversion factors (not available via exchange rate APIs)
_FIXED_CURRENCY_FACTORS = {
    "GBX": {"GBP": 0.01},  # pence to pounds
}


def fetch_exchange_rates(base_currency: str, currencies: set[str]) -> dict[str, float]:
    """Fetch exchange rates from the Frankfurter API (ECB data).

    Returns a dict mapping each requested currency to its conversion factor
    to the base currency.
    """
    if not currencies:
        return {}

    symbols = ",".join(sorted(currencies))
    url = (
        f"https://api.frankfurter.dev/v1/latest?base={base_currency}&symbols={symbols}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "tradey/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"could not fetch exchange rates from frankfurter.dev: {e}"
        ) from e

    rates = data.get("rates") if isinstance(data, dict) else None
    if not isinstance(rates, dict):
        raise RuntimeError(
            "could not fetch exchange rates from frankfurter.dev: "
            "response did not contain a 'rates' object"
        )

    # API returns rates as base→foreign, we need foreign→base (i.e. 1/rate)
    factors: dict[str, float] = {}
    for currency, rate in rates.items():
        if isinstance(rate, bool) or not isinstance(rate, int | float):
            raise RuntimeError(
                f"could not fetch exchange rates from frankfurter.dev: "
                f"non-numeric rate for {currency}: {rate!r}"
            )
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError(
                f"could not fetch exchange rates from frankfurter.dev: "
                f"invalid rate for {currency}: {rate!r}"
            )
        factors[currency] = 1.0 / rate
    return factors


def resolve_currency_factors(
    base_currency: str,
    assigned_currencies: set[str],
) -> dict[str, float]:
    """Map each currency to its conversion factor to the base currency."""
    factors: dict[str, float] = {base_currency: 1.0}
    for currency in assigned_currencies:
        fixed = _FIXED_CURRENCY_FACTORS.get(currency, {})
        if base_currency in fixed:
            factors[currency] = fixed[base_currency]

    need_fetch = assigned_currencies - set(factors)
    if need_fetch:
        factors.update(fetch_exchange_rates(base_currency, need_fetch))

    return factors
