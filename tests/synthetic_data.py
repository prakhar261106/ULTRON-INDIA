"""SYNTHETIC (fake) market data for unit tests and notebook demos ONLY.

!!! Everything produced here is random and has NO relation to real NSE prices. !!!
It must never be used to claim anything about market performance. Any result computed from it
must be labelled SYNTHETIC (the Stock Selection Engine takes ``data_label="SYNTHETIC..."`` and
adds a SYNTHETIC_DATA risk flag to every output).
"""
from __future__ import annotations

import math
import zlib
from datetime import date, datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from shared.contracts.market_data import CANDLE_COLUMNS, MarketDataAdapter, MarketDataError
from shared.contracts.types import IST, Timeframe

SYNTHETIC_LABEL = "SYNTHETIC_TEST_DATA"
SESSION_MINUTES = 375  # 09:15 - 15:30


def weekdays_ending(end_date: date, n: int) -> List[date]:
    """The last ``n`` weekdays up to and including ``end_date`` (no holiday logic)."""
    days, d = [], end_date
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def _bar_opens(day: date, tf: Timeframe) -> List[datetime]:
    start = datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    n = math.ceil(SESSION_MINUTES / tf.minutes)
    return [start + timedelta(minutes=k * tf.minutes) for k in range(n)]


def _rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def _build_frame(stamps, rets, gaps_first, vols, start_price, hl_noise) -> pd.DataFrame:
    n = len(stamps)
    opens, highs, lows, closes = (np.empty(n) for _ in range(4))
    prev_close = start_price
    for i in range(n):
        o = prev_close * math.exp(gaps_first[i])
        c = o * math.exp(rets[i])
        opens[i], closes[i] = o, c
        highs[i] = max(o, c) * math.exp(hl_noise[i, 0])
        lows[i] = min(o, c) * math.exp(-hl_noise[i, 1])
        prev_close = c
    return pd.DataFrame(
        {"timestamp": pd.DatetimeIndex(stamps), "open": opens, "high": highs, "low": lows,
         "close": closes, "volume": vols}
    )[CANDLE_COLUMNS]


def make_synthetic_market(
    symbols: Sequence[str],
    timeframe: Timeframe | str = "15m",
    n_sessions: int = 25,
    end_date: Optional[date] = None,
    seed: int = 42,
    sector_of: Optional[Mapping[str, str]] = None,
    base_volume: Optional[Mapping[str, float]] = None,
    base_price: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ``({symbol: candles}, index_candles)`` of fully synthetic data."""
    tf = Timeframe.parse(timeframe)
    end_date = end_date or date(2026, 11, 6)
    days = weekdays_ending(end_date, n_sessions)
    stamps = [t for d in days for t in _bar_opens(d, tf)]
    nb = math.ceil(SESSION_MINUTES / tf.minutes)
    n = len(stamps)
    first_of_day = np.array([i % nb == 0 for i in range(n)])
    sector_of = sector_of or {}

    # common market factor
    mrng = _rng(seed, "__market__")
    mkt = mrng.normal(0, 0.008 / math.sqrt(nb), n)
    sector_cache: Dict[str, np.ndarray] = {}

    def sector_factor(name: str) -> np.ndarray:
        if name not in sector_cache:
            sector_cache[name] = _rng(seed, "sector:" + name).normal(0, 0.004 / math.sqrt(nb), n)
        return sector_cache[name]

    profile = np.array([1 + 1.5 * math.exp(-k / 3) + 1.0 * math.exp(-(nb - 1 - k) / 3) for k in range(nb)])
    profile /= profile.sum()

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        r = _rng(seed, sym)
        beta = r.uniform(0.7, 1.3)
        idio = r.normal(0, r.uniform(0.006, 0.014) / math.sqrt(nb), n)
        rets = beta * mkt + sector_factor(sector_of.get(sym, "NONE")) + idio
        gaps = np.where(first_of_day, r.normal(0, 0.004, n), 0.0)
        daily_total = np.repeat((base_volume or {}).get(sym, r.uniform(1.5e6, 4e6)) * r.lognormal(0, 0.25, len(days)), nb)
        vols = np.round(daily_total * np.tile(profile, len(days)) * r.lognormal(0, 0.3, n))
        hl = np.abs(r.normal(0, 0.0008, (n, 2)))
        price = (base_price or {}).get(sym, r.uniform(300, 3000))
        out[sym] = _build_frame(stamps, rets, gaps, vols, price, hl)

    irng = _rng(seed, "__index__")
    index_df = _build_frame(
        stamps, mkt, np.where(first_of_day, irng.normal(0, 0.003, n), 0.0),
        np.zeros(n), 23000.0, np.abs(irng.normal(0, 0.0005, (n, 2))),
    )
    return out, index_df


class SyntheticMarketAdapter(MarketDataAdapter):
    """Adapter serving synthetic candles. For demos/tests only; offers NO live data."""

    name = "SYNTHETIC"
    adjusted_for_corporate_actions = True

    def __init__(self, symbols: Sequence[str], end_date: date, n_sessions: int = 25, seed: int = 42,
                 sector_of=None, base_volume=None, index_symbol: str = "NIFTY 50"):
        self._data: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._args = dict(symbols=list(symbols), end_date=end_date, n_sessions=n_sessions, seed=seed,
                          sector_of=sector_of, base_volume=base_volume)
        self.index_symbol = index_symbol

    def _ensure(self, tf: Timeframe):
        key = ("__built__", tf.value)
        if key not in self._data:
            stocks, idx = make_synthetic_market(timeframe=tf, **self._args)
            for s, d in stocks.items():
                self._data[(s, tf.value)] = d
            self._data[(self.index_symbol, tf.value)] = idx
            self._data[key] = pd.DataFrame()

    def get_historical_candles(self, symbol, timeframe, start, end):
        tf = Timeframe.parse(timeframe)
        self._ensure(tf)
        if (symbol, tf.value) not in self._data:
            raise MarketDataError(f"SYNTHETIC adapter has no data for {symbol}")
        df = self._data[(symbol, tf.value)]
        m = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
        return df.loc[m].reset_index(drop=True).copy()

    def get_live_quotes(self, symbols):
        raise MarketDataError("Synthetic data can never be live data.")
