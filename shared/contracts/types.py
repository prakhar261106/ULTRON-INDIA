"""Shared basic types for every ULTRON engine.

Contents
--------
* ``IST``              - the Asia/Kolkata timezone used for ALL timestamps.
* ``Timeframe``        - the supported candle sizes (5m, 15m, 1h).
* ``Direction`` etc.   - enumerations used in the engine output contract.
* ``normalize_symbol`` - strict NSE symbol normaliser.
"""
from __future__ import annotations

import re
from enum import Enum
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Timeframe(str, Enum):
    """Supported candle timeframes."""

    M5 = "5m"
    M15 = "15m"
    H1 = "1h"

    @property
    def minutes(self) -> int:
        return {"5m": 5, "15m": 15, "1h": 60}[self.value]

    @classmethod
    def parse(cls, value: "Timeframe | str") -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(t.value for t in cls)
            raise ValueError(f"Unsupported timeframe {value!r}. Allowed: {allowed}") from exc


class Direction(str, Enum):
    """Directional call of an engine."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    NO_TRADE = "NO_TRADE"


class DataQualityStatus(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class ModelStatus(str, Enum):
    """Maturity of whatever produced the output."""

    RULE_BASED = "RULE_BASED"  # deterministic rules, nothing was trained
    UNTRAINED = "UNTRAINED"
    TRAINED_UNVALIDATED = "TRAINED_UNVALIDATED"
    VALIDATED = "VALIDATED"  # passed out-of-sample / walk-forward checks
    CALIBRATED = "CALIBRATED"  # confidence validated as a probability
    DEGRADED = "DEGRADED"  # was validated, performance has since dropped


# NSE trading symbols may contain letters, digits, '&' and '-' (M&M, BAJAJ-AUTO).
# Index names contain a space (NIFTY 50, NIFTY BANK).
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\- ]{0,29}$")

INDEX_NIFTY50 = "NIFTY 50"
INDEX_NIFTY_BANK = "NIFTY BANK"


def normalize_symbol(symbol: str) -> str:
    """Return a clean upper-case NSE symbol or raise ``ValueError``.

    This is deliberately strict. Provider-specific forms such as ``NSE:TCS`` or
    ``TCS.NS`` are rejected, because converting them is the job of a provider
    adapter, not of the shared contract.
    """
    if not isinstance(symbol, str):
        raise TypeError(f"symbol must be a string, got {type(symbol).__name__}")
    cleaned = symbol.strip().upper()
    if not _SYMBOL_RE.match(cleaned):
        raise ValueError(
            f"Invalid symbol {symbol!r}. Use the plain NSE trading symbol, e.g. 'RELIANCE', "
            "'M&M', 'BAJAJ-AUTO' or an index name such as 'NIFTY 50'."
        )
    return cleaned
