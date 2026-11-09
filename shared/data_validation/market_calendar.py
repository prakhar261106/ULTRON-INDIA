"""NSE session helpers: trading days, bar grids, bar-close times, trading-time arithmetic.

Limits (be aware): ULTRON has no official holiday feed. Holidays come from
``config.market.holidays``. Special sessions (e.g. Diwali Muhurat trading, weekend special
sessions) are NOT treated as normal trading days; candles in them are reported as warnings.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List

import pandas as pd

from config.settings import MarketConfig
from shared.contracts.types import IST, Timeframe


def is_trading_day(d: date, market: MarketConfig) -> bool:
    return d.weekday() < 5 and d not in set(market.holidays)


def session_open_dt(d: date, market: MarketConfig) -> datetime:
    return datetime.combine(d, market.session_open, tzinfo=IST)


def session_close_dt(d: date, market: MarketConfig) -> datetime:
    return datetime.combine(d, market.session_close, tzinfo=IST)


def grid_bar_opens(d: date, timeframe: Timeframe, market: MarketConfig) -> List[datetime]:
    """Expected bar OPEN times for one session, anchored at the session open.

    1h bars therefore open 09:15, 10:15 ... 15:15 (the last one is only 15 minutes long).
    """
    opens, t = [], session_open_dt(d, market)
    end = session_close_dt(d, market)
    step = timedelta(minutes=timeframe.minutes)
    while t < end:
        opens.append(t)
        t += step
    return opens


def bar_close_times(timestamps: pd.Series, timeframe: Timeframe, market: MarketConfig) -> pd.Series:
    """Close time of each bar (``open + timeframe``, capped at the session close)."""
    ts = timestamps.dt.tz_convert(IST)
    raw = ts + pd.Timedelta(minutes=timeframe.minutes)
    day_close = ts.dt.normalize() + pd.Timedelta(
        hours=market.session_close.hour, minutes=market.session_close.minute
    )
    cap = (ts < day_close) & (raw > day_close)
    return raw.where(~cap, day_close)


def trading_minutes_between(start: datetime, end: datetime, market: MarketConfig) -> float:
    """Minutes of NSE trading time between two instants (overnight/weekends/holidays count 0)."""
    start, end = start.astimezone(IST), end.astimezone(IST)
    if end <= start:
        return 0.0
    total, d = 0.0, start.date()
    while d <= end.date():
        if is_trading_day(d, market):
            lo = max(start, session_open_dt(d, market))
            hi = min(end, session_close_dt(d, market))
            if hi > lo:
                total += (hi - lo).total_seconds() / 60.0
        d += timedelta(days=1)
    return total
