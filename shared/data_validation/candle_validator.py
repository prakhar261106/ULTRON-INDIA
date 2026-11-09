"""Validation of OHLCV candle data.

``validate_candles`` NEVER changes the data and never "fixes" problems silently. It returns a
``ValidationReport`` listing every issue. Engines exclude data that has ERROR issues.

Checks
------
ERROR   EMPTY_DATA, MISSING_COLUMNS, BAD_TIMESTAMP_DTYPE, NAIVE_TIMESTAMP, NON_NUMERIC,
        MISSING_VALUES, NON_FINITE_VALUES, DUPLICATE_CANDLES, UNSORTED_TIMESTAMPS,
        NON_POSITIVE_PRICE, INVALID_OHLC, NEGATIVE_VOLUME, STALE_DATA, MISSING_CANDLES (> limit)
WARNING MISSING_CANDLES (<= limit), MISSING_SESSION_DAYS, OFF_SESSION_BARS, MISALIGNED_BARS,
        FUTURE_TIMESTAMPS, ZERO_VOLUME_BARS

Candle timestamps are bar OPEN times. A bar is "complete" once its close time <= ``as_of``;
the still-forming bar is ignored (counted in ``n_incomplete_bars``), never validated as data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config.settings import DataQualityConfig, MarketConfig
from shared.contracts.engine_output import DataQuality
from shared.contracts.market_data import CANDLE_COLUMNS
from shared.contracts.types import IST, DataQualityStatus, Timeframe
from shared.data_validation.market_calendar import (
    bar_close_times,
    grid_bar_opens,
    is_trading_day,
    session_close_dt,
    trading_minutes_between,
)

PRICE_COLS = ["open", "high", "low", "close"]


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    severity: Severity
    message: str
    count: int = 0


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = ""
    timeframe: Timeframe
    as_of: datetime
    n_rows: int = 0
    n_incomplete_bars: int = 0
    last_complete_bar_open: Optional[datetime] = None
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def codes(self) -> List[str]:
        return [i.code for i in self.issues]

    @property
    def quality_score(self) -> float:
        return float(max(0.0, 1.0 - 0.5 * len(self.errors) - 0.1 * len(self.warnings)))

    @property
    def status(self) -> DataQualityStatus:
        if self.errors:
            return DataQualityStatus.POOR
        return DataQualityStatus.DEGRADED if self.warnings else DataQualityStatus.GOOD

    def to_data_quality(self) -> DataQuality:
        return DataQuality(
            status=self.status,
            score=self.quality_score,
            n_observations=self.n_rows,
            last_observation_time=self.last_complete_bar_open,
            issues=[f"{i.severity.value}:{i.code}" for i in self.issues],
        )


def validate_candles(
    df: pd.DataFrame,
    timeframe: Timeframe | str,
    as_of: datetime,
    market: MarketConfig | None = None,
    quality: DataQualityConfig | None = None,
    symbol: str = "",
) -> ValidationReport:
    """Validate a candle DataFrame against the ULTRON candle format."""
    market = market or MarketConfig()
    quality = quality or DataQualityConfig()
    tf = Timeframe.parse(timeframe)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(IST)

    report = ValidationReport(symbol=symbol, timeframe=tf, as_of=as_of)

    def err(code, msg, count=0):
        report.issues.append(ValidationIssue(code=code, severity=Severity.ERROR, message=msg, count=count))

    def warn(code, msg, count=0):
        report.issues.append(ValidationIssue(code=code, severity=Severity.WARNING, message=msg, count=count))

    # ---- structure -------------------------------------------------------------------
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        err("EMPTY_DATA", "No candles supplied.")
        return report
    report.n_rows = len(df)
    missing_cols = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing_cols:
        err("MISSING_COLUMNS", f"Missing columns: {missing_cols}", len(missing_cols))
        return report
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        err("BAD_TIMESTAMP_DTYPE", "timestamp column must be a datetime column.")
        return report
    if df["timestamp"].dt.tz is None:
        err("NAIVE_TIMESTAMP", "timestamps have no timezone; ULTRON never guesses the timezone.")
        return report
    non_numeric = [c for c in PRICE_COLS + ["volume"] if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        err("NON_NUMERIC", f"Non-numeric columns: {non_numeric}", len(non_numeric))
        return report

    # ---- missing / non-finite values ---------------------------------------------------
    na_rows = df[CANDLE_COLUMNS].isna().any(axis=1)
    if na_rows.any():
        err("MISSING_VALUES", "Rows contain missing values (NaN/NaT).", int(na_rows.sum()))
    clean = df.loc[~na_rows].copy()
    num = clean[PRICE_COLS + ["volume"]].to_numpy(dtype=float)
    inf_rows = ~np.isfinite(num).all(axis=1)
    if inf_rows.any():
        err("NON_FINITE_VALUES", "Rows contain infinite values.", int(inf_rows.sum()))
        clean = clean.loc[~inf_rows]
    if clean.empty:
        return report

    # ---- ordering / duplicates ---------------------------------------------------------
    ts = clean["timestamp"].dt.tz_convert(IST)
    dup = ts.duplicated()
    if dup.any():
        err("DUPLICATE_CANDLES", "Duplicate candle timestamps found.", int(dup.sum()))
    if not ts.is_monotonic_increasing:
        err("UNSORTED_TIMESTAMPS", "Timestamps are not sorted ascending.")

    # ---- price sanity ------------------------------------------------------------------
    o, h, l, c, v = (clean[x].astype(float) for x in ["open", "high", "low", "close", "volume"])
    nonpos = (clean[PRICE_COLS] <= 0).any(axis=1)
    if nonpos.any():
        err("NON_POSITIVE_PRICE", "Prices must be > 0.", int(nonpos.sum()))
    bad_ohlc = (h < l) | (h < pd.concat([o, c], axis=1).max(axis=1)) | (l > pd.concat([o, c], axis=1).min(axis=1))
    if bad_ohlc.any():
        err("INVALID_OHLC", "Inconsistent OHLC (high < low/open/close or low > open/close).", int(bad_ohlc.sum()))
    if (v < 0).any():
        err("NEGATIVE_VOLUME", "Volume cannot be negative.", int((v < 0).sum()))

    # ---- completeness vs as_of ---------------------------------------------------------
    closes = bar_close_times(ts, tf, market)
    future_open = ts > as_of
    if future_open.any():
        warn("FUTURE_TIMESTAMPS", "Candles opening after as_of (possible look-ahead data).", int(future_open.sum()))
    complete = closes <= as_of
    report.n_incomplete_bars = int((~complete & ~future_open).sum())
    ts_c = ts[complete]
    if ts_c.empty:
        err("EMPTY_DATA", "No completed candles at as_of.")
        return report
    report.last_complete_bar_open = ts_c.max().to_pydatetime()

    # ---- session alignment -------------------------------------------------------------
    open_min = market.session_open.hour * 60 + market.session_open.minute
    mins = ts_c.dt.hour * 60 + ts_c.dt.minute - open_min
    in_session = (mins >= 0) & (mins < market.session_minutes) & (ts_c.dt.weekday < 5)
    if (~in_session).any():
        warn("OFF_SESSION_BARS", "Candles outside the normal NSE session (special session?).", int((~in_session).sum()))
    misaligned = in_session & (mins % tf.minutes != 0)
    if misaligned.any():
        warn("MISALIGNED_BARS", f"Candles not on the {tf.value} grid anchored at the session open.", int(misaligned.sum()))

    # ---- missing candles / days --------------------------------------------------------
    ongrid = ts_c[in_session & (mins % tf.minutes == 0)]
    observed_by_day: dict = {}
    for stamp in ongrid:
        observed_by_day.setdefault(stamp.date(), set()).add(stamp.to_pydatetime())
    expected_total = missing_total = 0
    for d, observed in observed_by_day.items():
        if not is_trading_day(d, market):
            continue
        grid = grid_bar_opens(d, tf, market)
        if d == as_of.date():
            grid = [g for g in grid if min(g + timedelta(minutes=tf.minutes), session_close_dt(d, market)) <= as_of]
        expected_total += len(grid)
        missing_total += len([g for g in grid if g not in observed])
    if expected_total:
        ratio = missing_total / expected_total
        if missing_total and ratio > quality.max_missing_candle_ratio:
            err("MISSING_CANDLES", f"{ratio:.1%} of expected candles are missing.", missing_total)
        elif missing_total:
            warn("MISSING_CANDLES", f"{ratio:.1%} of expected candles are missing.", missing_total)

    days_present = set(ts_c.dt.date)
    d, last_day, first_day, absent = min(days_present), max(days_present), min(days_present), 0
    while d <= last_day:
        if is_trading_day(d, market) and d not in days_present:
            absent += 1
        d += timedelta(days=1)
    if absent > quality.max_unexplained_missing_days:
        warn("MISSING_SESSION_DAYS",
             f"{absent} weekdays have no candles (undeclared holidays? add them to config.market.holidays).", absent)

    # ---- staleness ---------------------------------------------------------------------
    last_close = closes[complete].max().to_pydatetime()
    stale_minutes = trading_minutes_between(last_close, as_of, market)
    limit = quality.max_staleness_bars * tf.minutes
    if stale_minutes > limit:
        err("STALE_DATA", f"Newest completed candle closed {stale_minutes:.0f} trading minutes before as_of (limit {limit}).",
            int(stale_minutes))

    # ---- zero volume -------------------------------------------------------------------
    zero_ratio = float((v.loc[ts_c.index] == 0).mean())
    if zero_ratio > quality.max_zero_volume_ratio:
        warn("ZERO_VOLUME_BARS", f"{zero_ratio:.1%} of candles have zero volume.", int((v.loc[ts_c.index] == 0).sum()))
    return report


def complete_bars(df: pd.DataFrame, timeframe: Timeframe | str, as_of: datetime, market: MarketConfig | None = None) -> pd.DataFrame:
    """Return only candles whose bar has CLOSED by ``as_of`` (the look-ahead guard)."""
    market = market or MarketConfig()
    tf = Timeframe.parse(timeframe)
    as_of = as_of.astimezone(IST)
    closes = bar_close_times(df["timestamp"], tf, market)
    return df.loc[closes <= as_of].reset_index(drop=True)
