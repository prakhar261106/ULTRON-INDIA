"""Provider-independent market-data interface for ULTRON.

NO broker or data vendor is implemented here. A provider (Dhan, Zerodha, a CSV export,
...) is added later by writing a small subclass of :class:`MarketDataAdapter`.

Data formats (the contract every adapter must follow)
-----------------------------------------------------
Symbol     Plain NSE trading symbol, upper case: ``RELIANCE``, ``M&M``, ``BAJAJ-AUTO``.
           Indices: ``NIFTY 50``, ``NIFTY BANK``. Provider-specific forms (``NSE:TCS``,
           ``TCS.NS``) must be converted INSIDE the adapter.
Candles    ``pandas.DataFrame`` with exactly the columns
           ``timestamp, open, high, low, close, volume``.
           * ``timestamp`` = candle OPEN time, timezone-aware, converted to Asia/Kolkata.
           * sorted ascending, no duplicates, float prices, non-negative volume.
           * prices adjusted for splits/bonus issues (``adjusted_for_corporate_actions``
             says whether the adapter guarantees this; ``None`` means unknown).
Live data  ``LiveQuote`` objects. A quote is only "live" if its delay is KNOWN and small.
           Unknown delay = NOT live. Delayed, stale or simulated data is never live.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.contracts.types import IST, Timeframe, normalize_symbol

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MarketDataError(RuntimeError):
    """Raised when market data cannot be obtained or is malformed."""


class ProviderNotConfiguredError(MarketDataError):
    """Raised when no market-data provider has been configured yet."""


class LiveQuote(BaseModel):
    """A single live (or claimed-live) price observation."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    exchange_timestamp: datetime = Field(description="Time stamped by the exchange/provider.")
    received_at: datetime = Field(description="Time ULTRON received the quote.")
    last_price: float = Field(gt=0, allow_inf_nan=False)
    volume: Optional[float] = Field(default=None, ge=0)
    data_delay_seconds: Optional[float] = Field(
        default=None, ge=0, description="Known provider delay. None = unknown (treated as NOT live)."
    )

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v):
        return normalize_symbol(v)

    @field_validator("exchange_timestamp", "received_at")
    @classmethod
    def _aware(cls, v):
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v.astimezone(IST)


def is_quote_live(
    quote: LiveQuote, now: datetime, max_age_seconds: float = 5.0, max_delay_seconds: float = 2.0
) -> bool:
    """True only if the delay is known and small AND the quote is recent.

    Unknown delay, delayed feeds and old quotes all return False.
    """
    if quote.data_delay_seconds is None or quote.data_delay_seconds > max_delay_seconds:
        return False
    age = (now.astimezone(IST) - quote.exchange_timestamp).total_seconds()
    return 0 <= age <= max_age_seconds


class MarketDataAdapter(ABC):
    """Interface every market-data provider must implement."""

    name: str = "abstract"
    adjusted_for_corporate_actions: Optional[bool] = None

    @abstractmethod
    def get_historical_candles(
        self, symbol: str, timeframe: Timeframe | str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return candles with open time in ``[start, end]`` in the standard format."""

    @abstractmethod
    def get_live_quotes(self, symbols: Sequence[str]) -> List[LiveQuote]:
        """Return the latest quotes for ``symbols``."""


class UnconfiguredAdapter(MarketDataAdapter):
    """Default adapter: refuses to work and explains what is missing.

    It exists so that the system can NEVER silently fall back to fake data.
    """

    name = "unconfigured"

    _MESSAGE = (
        "No market-data provider is configured. Choose a provider, add its credentials via "
        "environment variables / Colab Secrets, and implement a MarketDataAdapter subclass "
        "(see shared/contracts/market_data.py). ULTRON will not substitute fake data."
    )

    def get_historical_candles(self, symbol, timeframe, start, end):
        raise ProviderNotConfiguredError(self._MESSAGE)

    def get_live_quotes(self, symbols):
        raise ProviderNotConfiguredError(self._MESSAGE)


class CsvMarketDataAdapter(MarketDataAdapter):
    """Reads candles from CSV files you exported yourself (any provider).

    File name: ``<SYMBOL>_<timeframe>.csv`` with spaces in the symbol replaced by ``_``,
    e.g. ``RELIANCE_15m.csv`` or ``NIFTY_50_15m.csv``.
    Columns: ``timestamp,open,high,low,close,volume``.

    Timestamps with a UTC offset (``2026-11-06 09:15:00+05:30``) are converted to IST.
    Timestamps WITHOUT an offset are rejected unless ``naive_timestamps_tz`` is given -
    the adapter will not guess your timezone.
    """

    name = "csv"

    def __init__(
        self,
        directory: str | Path,
        naive_timestamps_tz: Optional[str] = None,
        prices_adjusted: Optional[bool] = None,
    ):
        self.directory = Path(directory)
        self.naive_timestamps_tz = naive_timestamps_tz
        self.adjusted_for_corporate_actions = prices_adjusted

    def _path(self, symbol: str, timeframe: Timeframe) -> Path:
        stem = normalize_symbol(symbol).replace(" ", "_")
        return self.directory / f"{stem}_{timeframe.value}.csv"

    def get_historical_candles(self, symbol, timeframe, start, end) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        path = self._path(symbol, tf)
        if not path.exists():
            raise MarketDataError(f"CSV file not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
        if missing:
            raise MarketDataError(f"{path.name} is missing columns: {missing}")
        df = df[CANDLE_COLUMNS].copy()
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any():
            raise MarketDataError(f"{path.name}: unparseable or mixed-offset timestamps")
        if ts.dt.tz is None:
            if not self.naive_timestamps_tz:
                raise MarketDataError(
                    f"{path.name}: timestamps have no timezone. Add a UTC offset to the file or "
                    "construct the adapter with naive_timestamps_tz='Asia/Kolkata'."
                )
            ts = ts.dt.tz_localize(self.naive_timestamps_tz)
        df["timestamp"] = ts.dt.tz_convert(IST)
        mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
        return df.loc[mask].reset_index(drop=True)

    def get_live_quotes(self, symbols):
        raise MarketDataError("The CSV adapter has no live data. Configure a live provider adapter.")
