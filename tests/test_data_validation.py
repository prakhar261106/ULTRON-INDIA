"""Tests for candle validation (valid data, missing values, duplicates, invalid OHLC, stale data, ...)."""
from datetime import datetime

import numpy as np
import pandas as pd

from shared.contracts.types import IST, DataQualityStatus
from shared.data_validation import complete_bars, trading_minutes_between, validate_candles
from config.settings import MarketConfig
from tests.synthetic_data import make_synthetic_market
from tests.conftest import AS_OF, END_DATE


def codes(report):
    return set(report.codes)


def test_valid_market_data_passes(candles, as_of):
    rep = validate_candles(candles, "15m", as_of, symbol="AAA")
    assert rep.is_valid and rep.codes == []
    assert rep.status == DataQualityStatus.GOOD and rep.quality_score == 1.0


def test_missing_values_detected(candles, as_of):
    candles.loc[5, "close"] = np.nan
    rep = validate_candles(candles, "15m", as_of)
    assert "MISSING_VALUES" in codes(rep) and not rep.is_valid
    assert rep.status == DataQualityStatus.POOR


def test_duplicate_candles_detected(candles, as_of):
    dup = pd.concat([candles, candles.iloc[[10]]]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    rep = validate_candles(dup, "15m", as_of)
    assert "DUPLICATE_CANDLES" in codes(rep) and not rep.is_valid


def test_invalid_ohlc_detected(candles, as_of):
    candles.loc[3, "high"] = candles.loc[3, "low"] - 1.0
    rep = validate_candles(candles, "15m", as_of)
    assert "INVALID_OHLC" in codes(rep) and not rep.is_valid


def test_non_positive_price_and_negative_volume(candles, as_of):
    candles.loc[2, "close"] = 0.0
    candles.loc[4, "volume"] = -5.0
    rep = validate_candles(candles, "15m", as_of)
    assert {"NON_POSITIVE_PRICE", "NEGATIVE_VOLUME"} <= codes(rep)


def test_unsorted_timestamps_detected(candles, as_of):
    shuffled = candles.iloc[[1, 0] + list(range(2, len(candles)))].reset_index(drop=True)
    assert "UNSORTED_TIMESTAMPS" in codes(validate_candles(shuffled, "15m", as_of))


def test_stale_timestamps_detected(candles):
    monday_midday = datetime(2026, 11, 9, 12, 0, tzinfo=IST)  # data ends Friday 15:30
    rep = validate_candles(candles, "15m", monday_midday)
    assert "STALE_DATA" in codes(rep) and not rep.is_valid


def test_weekend_gap_is_not_stale(candles):
    monday_open = datetime(2026, 11, 9, 9, 20, tzinfo=IST)  # only 5 trading minutes after Friday close
    rep = validate_candles(candles, "15m", monday_open)
    assert "STALE_DATA" not in codes(rep)


def test_naive_timestamps_rejected(candles, as_of):
    candles["timestamp"] = candles["timestamp"].dt.tz_localize(None)
    rep = validate_candles(candles, "15m", as_of)
    assert "NAIVE_TIMESTAMP" in codes(rep) and not rep.is_valid


def test_missing_columns_and_empty(candles, as_of):
    assert "MISSING_COLUMNS" in codes(validate_candles(candles.drop(columns=["volume"]), "15m", as_of))
    assert "EMPTY_DATA" in codes(validate_candles(candles.iloc[0:0], "15m", as_of))
    assert "EMPTY_DATA" in codes(validate_candles(None, "15m", as_of))


def test_few_missing_candles_is_warning_many_is_error(candles, as_of):
    few = candles.drop(index=[40, 41, 42, 43, 44]).reset_index(drop=True)
    rep = validate_candles(few, "15m", as_of)
    assert rep.is_valid and "MISSING_CANDLES" in codes(rep)
    assert rep.status == DataQualityStatus.DEGRADED
    many = candles.drop(index=list(range(0, len(candles), 5))).reset_index(drop=True)  # 20% missing
    assert not validate_candles(many, "15m", as_of).is_valid


def test_misaligned_bars_flagged(candles, as_of):
    rep = validate_candles(candles, "1h", as_of)  # 15m data labelled as 1h
    assert "MISALIGNED_BARS" in codes(rep)


def test_future_and_incomplete_bars(candles):
    midday = datetime(2026, 11, 6, 12, 0, tzinfo=IST)
    rep = validate_candles(candles, "15m", midday)
    assert "FUTURE_TIMESTAMPS" in codes(rep)
    assert rep.n_incomplete_bars == 1  # the bar opening exactly at 12:00 is still forming


def test_complete_bars_is_the_lookahead_guard(candles):
    midday = datetime(2026, 11, 6, 12, 0, tzinfo=IST)
    kept = complete_bars(candles, "15m", midday)
    last_open = kept["timestamp"].iloc[-1]
    assert last_open == pd.Timestamp("2026-11-06 11:45", tz=IST)


def test_last_hourly_bar_closes_at_session_close():
    stocks, _ = make_synthetic_market(["AAA"], "1h", 12, END_DATE)
    kept = complete_bars(stocks["AAA"], "1h", AS_OF)
    assert len(kept) == len(stocks["AAA"])  # the 15:15 bar closes at 15:30, not 16:15


def test_trading_minutes_skip_nights_and_weekends():
    m = MarketConfig()
    fri = datetime(2026, 11, 6, 15, 0, tzinfo=IST)
    mon = datetime(2026, 11, 9, 9, 30, tzinfo=IST)
    assert trading_minutes_between(fri, mon, m) == 45.0
