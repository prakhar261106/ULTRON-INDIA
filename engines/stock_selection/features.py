"""Per-symbol feature calculation for the Stock Selection Engine.

Input candles must already be VALIDATED and contain COMPLETE bars only (see
``shared.data_validation.complete_bars``). Nothing here looks past the last candle, so no
look-ahead is possible inside this module.

Features (all computed per symbol; ``signed`` values are kept for evidence)
--------------------------------------------------------------------------
avg_daily_traded_value  mean over prior sessions of sum(typical_price * volume)   [INR, approximate]
liquidity_bar_value     median over prior-session bars of typical_price * volume  [INR per bar]
relative_volume         volume so far in the latest session / average volume of prior sessions
                        over the SAME time-of-day bars
atr_pct                 ATR(period) / last close * 100 (true range computed inside each session)
volatility_pct          std of within-session log returns, scaled to a daily equivalent, in %
gap_pct                 (latest session open / previous session close - 1) * 100         [signed]
momentum_pct            % change of close over ``momentum_lookback_bars`` bars           [signed]
relative_strength_pct   momentum_pct minus NIFTY 50 momentum over the same two timestamps [signed]
sector_strength_pct     filled in by the engine (mean relative strength of the sector)  [signed]

"Prior sessions" = the ``lookback_sessions`` sessions BEFORE the latest session; the latest
session is never mixed into its own averages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from config.settings import MarketConfig, StockSelectionConfig
from shared.contracts.types import IST, Timeframe

FEATURE_NAMES = [
    "avg_daily_traded_value",
    "liquidity_bar_value",
    "relative_volume",
    "atr_pct",
    "volatility_pct",
    "gap_pct",
    "momentum_pct",
    "relative_strength_pct",
    "sector_strength_pct",
]
# Direction-agnostic ranking: size of the move matters, not its sign.
ABS_FEATURES = {"gap_pct", "momentum_pct", "relative_strength_pct", "sector_strength_pct"}

NAN = float("nan")


@dataclass
class SymbolFeatures:
    symbol: str
    values: Dict[str, float]
    last_close: float
    avg_daily_volume: float
    current_session: date
    n_prior_sessions: int
    notes: List[str] = field(default_factory=list)
    # feature name -> reason code, for every feature that could NOT be computed.
    # An unavailable feature is NaN (never 0) and always has an entry here.
    missing: Dict[str, str] = field(default_factory=dict)


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def compute_symbol_features(
    candles: pd.DataFrame,
    index_candles: Optional[pd.DataFrame],
    timeframe: Timeframe | str,
    cfg: StockSelectionConfig,
    market: MarketConfig,
    symbol: str = "",
) -> SymbolFeatures:
    tf = Timeframe.parse(timeframe)
    df = candles.sort_values("timestamp").reset_index(drop=True).copy()
    ts = df["timestamp"].dt.tz_convert(IST)
    open_min = market.session_open.hour * 60 + market.session_open.minute
    df["date"] = ts.dt.date
    df["k"] = ((ts.dt.hour * 60 + ts.dt.minute - open_min) // tf.minutes).astype(int)
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tv"] = df["typ"] * df["volume"]

    sessions = sorted(df["date"].unique())
    cur = sessions[-1]
    prior_sessions = sessions[:-1][-cfg.lookback_sessions :]
    values: Dict[str, float] = {name: NAN for name in FEATURE_NAMES}
    missing: Dict[str, str] = {}
    notes: List[str] = []
    last_close = _f(df["close"].iloc[-1])

    prior = df[df["date"].isin(prior_sessions)]
    cur_df = df[df["date"] == cur]

    avg_daily_volume = NAN
    if len(prior_sessions) > 0:
        daily = prior.groupby("date").agg(vol=("volume", "sum"), tv=("tv", "sum"))
        avg_daily_volume = _f(daily["vol"].mean())
        values["avg_daily_traded_value"] = _f(daily["tv"].mean())
        values["liquidity_bar_value"] = _f(prior["tv"].median())

        # relative volume: same time-of-day window in prior sessions
        kmax = int(cur_df["k"].max())
        prior_cum = prior[prior["k"] <= kmax].groupby("date")["volume"].sum()
        if len(prior_cum) > 0 and prior_cum.mean() > 0:
            values["relative_volume"] = _f(cur_df["volume"].sum() / prior_cum.mean())
        else:
            missing["relative_volume"] = "NO_PRIOR_VOLUME_AT_THIS_TIME_OF_DAY"

        # gap vs previous session close
        prev_day = sessions[-2]
        prev_close = float(df.loc[df["date"] == prev_day, "close"].iloc[-1])
        if prev_close > 0:
            values["gap_pct"] = _f((float(cur_df["open"].iloc[0]) / prev_close - 1.0) * 100.0)
    else:
        notes.append("no prior sessions available")
        for name in ("avg_daily_traded_value", "liquidity_bar_value", "relative_volume", "gap_pct"):
            missing[name] = "NO_PRIOR_SESSIONS"

    # ATR% with true range computed inside each session (no overnight gap contamination)
    prev_close_in_session = df.groupby("date")["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close_in_session).abs(), (df["low"] - prev_close_in_session).abs()],
        axis=1,
    ).max(axis=1)
    if len(df) >= cfg.atr_period and last_close > 0:
        values["atr_pct"] = _f(tr.rolling(cfg.atr_period).mean().iloc[-1] / last_close * 100.0)
    else:
        missing["atr_pct"] = "INSUFFICIENT_BARS"

    # volatility: within-session log returns over the lookback window, daily-equivalent
    window = df[df["date"].isin(list(prior_sessions) + [cur])]
    lr = np.log(window["close"] / window.groupby("date")["close"].shift(1)).dropna()
    if len(lr) >= 10:
        bars_per_day = market.session_minutes / tf.minutes
        values["volatility_pct"] = _f(lr.std(ddof=1) * math.sqrt(bars_per_day) * 100.0)
    else:
        missing["volatility_pct"] = "INSUFFICIENT_RETURNS"

    # momentum and relative strength vs index over the SAME two timestamps
    n = cfg.momentum_lookback_bars
    if len(df) > n:
        now_ts, then_ts = ts.iloc[-1], ts.iloc[-1 - n]
        values["momentum_pct"] = _f((df["close"].iloc[-1] / df["close"].iloc[-1 - n] - 1.0) * 100.0)
        if index_candles is not None and len(index_candles) > 0:
            idx = index_candles.drop_duplicates("timestamp").set_index(index_candles["timestamp"].dt.tz_convert(IST))["close"]
            pair = idx.reindex([now_ts, then_ts])
            if pair.notna().all() and pair.iloc[1] > 0:
                idx_mom = (pair.iloc[0] / pair.iloc[1] - 1.0) * 100.0
                values["relative_strength_pct"] = _f(values["momentum_pct"] - idx_mom)
            else:
                missing["relative_strength_pct"] = "INDEX_BARS_NOT_ALIGNED"
        else:
            missing["relative_strength_pct"] = "NO_INDEX_DATA"
    else:
        missing["momentum_pct"] = "INSUFFICIENT_BARS"
        missing["relative_strength_pct"] = "INSUFFICIENT_BARS"

    # Computed later, once every symbol's relative strength is known (needs universe context).
    missing["sector_strength_pct"] = "PENDING_UNIVERSE_CONTEXT"
    # Safety net: any NaN feature must carry a reason - nothing is ever missing silently.
    for name, val in values.items():
        if math.isnan(val) and name not in missing:
            missing[name] = "UNAVAILABLE"

    return SymbolFeatures(
        symbol=symbol,
        values=values,
        last_close=last_close,
        avg_daily_volume=avg_daily_volume,
        current_session=cur,
        n_prior_sessions=len(prior_sessions),
        notes=notes,
        missing=missing,
    )


def apply_sector_strength(feats: Dict[str, SymbolFeatures], sector_of: Mapping[str, str]) -> None:
    """Fill ``sector_strength_pct`` in place, or record WHY it is unavailable.

    sector_strength_pct = mean signed relative strength of the symbol's sector members that
    have a relative strength value (the symbol itself included). It uses UNIVERSE members only;
    no sector INDEX data is involved (sector indices arrive with the Market Correlation engine).

    Reasons recorded when the value cannot be computed (the value stays NaN, never 0):
      SECTOR_UNKNOWN                 symbol has no sector label
      NO_SECTOR_RELATIVE_STRENGTH    no member of the sector has a relative strength value
                                     (typically: NIFTY 50 candles missing)
    """
    by_sector: Dict[str, List[float]] = {}
    for sym, f in feats.items():
        rs = f.values["relative_strength_pct"]
        sec = sector_of.get(sym, "UNKNOWN")
        if sec != "UNKNOWN" and not math.isnan(rs):
            by_sector.setdefault(sec, []).append(rs)
    for sym, f in feats.items():
        sec = sector_of.get(sym, "UNKNOWN")
        if sec == "UNKNOWN":
            f.values["sector_strength_pct"] = NAN
            f.missing["sector_strength_pct"] = "SECTOR_UNKNOWN"
        elif sec in by_sector:
            f.values["sector_strength_pct"] = float(sum(by_sector[sec]) / len(by_sector[sec]))
            f.missing.pop("sector_strength_pct", None)
        else:
            f.values["sector_strength_pct"] = NAN
            f.missing["sector_strength_pct"] = "NO_SECTOR_RELATIVE_STRENGTH"
