"""Tests for the Stock Selection Engine. All market data here is SYNTHETIC."""
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from config.universe import Universe, UniverseMember, dedupe_symbols, load_universe
from engines.stock_selection import StockSelectionEngine
from engines.stock_selection.features import compute_symbol_features
from engines.stock_selection.ranking import rank_candidates
from shared.contracts import (
    EngineOutput, IST, ModelStatus, Direction, ProviderNotConfiguredError, UnconfiguredAdapter,
)
from shared.data_validation import complete_bars
from tests.conftest import AS_OF, END_DATE, synthetic_universe_market
from tests.synthetic_data import SyntheticMarketAdapter, _bar_opens, weekdays_ending

LABEL = "SYNTHETIC_TEST_DATA"


def make_universe(symbols, sector="S"):
    return Universe(as_of=END_DATE, membership_history_verified=True,
                    members=[UniverseMember(symbol=s, sector=sector, indices=["NIFTY50"]) for s in symbols])


def names(n):
    return [f"STK{i:02d}" for i in range(n)]


@pytest.fixture
def engine(config):
    return StockSelectionEngine(config)


# ----------------------------------------------------------- ranking calculation ---
def test_ranking_calculation_matches_hand_computation():
    feats = pd.DataFrame({"relative_volume": [3.0, 2.0, 1.0], "atr_pct": [3.0, 1.0, 2.0]}, index=["A", "B", "C"])
    feats.index.name = "symbol"
    out = rank_candidates(feats, {"relative_volume": 0.75, "atr_pct": 0.25}).set_index("symbol")
    assert out.loc["A", "composite_score"] == pytest.approx(1.0)
    assert out.loc["B", "composite_score"] == pytest.approx(0.75 * 2 / 3 + 0.25 * 1 / 3)
    assert out.loc["C", "composite_score"] == pytest.approx(0.75 * 1 / 3 + 0.25 * 2 / 3)
    assert list(out.sort_values("rank").index) == ["A", "B", "C"]


def test_signed_features_ranked_by_absolute_size():
    feats = pd.DataFrame({"gap_pct": [-5.0, 1.0, 2.0]}, index=["A", "B", "C"])
    feats.index.name = "symbol"
    out = rank_candidates(feats, {"gap_pct": 1.0})
    assert list(out["symbol"]) == ["A", "C", "B"]  # |-5| > |2| > |1|


def test_missing_feature_renormalises_weights_and_lowers_coverage():
    feats = pd.DataFrame({"relative_volume": [3.0, 2.0, 1.0], "atr_pct": [3.0, np.nan, 2.0]}, index=["A", "B", "C"])
    feats.index.name = "symbol"
    out = rank_candidates(feats, {"relative_volume": 0.5, "atr_pct": 0.5}).set_index("symbol")
    assert out.loc["B", "feature_coverage"] == pytest.approx(0.5)
    assert out.loc["B", "composite_score"] == pytest.approx(2 / 3)  # only relative_volume counts


def test_ties_are_broken_alphabetically_and_deterministically():
    feats = pd.DataFrame({"relative_volume": [1.0, 1.0, 1.0]}, index=["C", "A", "B"])
    feats.index.name = "symbol"
    out = rank_candidates(feats, {"relative_volume": 1.0})
    assert list(out["symbol"]) == ["A", "B", "C"]


# -------------------------------------------------------------- feature maths ---
def _flat_frame(n_sessions=15, tf="15m", price=100.0, volume=100.0):
    from shared.contracts import Timeframe
    rows = []
    for d in weekdays_ending(END_DATE, n_sessions):
        for t in _bar_opens(d, Timeframe.parse(tf)):
            rows.append((t, price, price, price, price, volume))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_features_on_flat_data_are_exactly_known(config):
    df = complete_bars(_flat_frame(), "15m", datetime(2026, 11, 6, 12, 0, tzinfo=IST), config.market)
    f = compute_symbol_features(df, None, "15m", config.stock_selection, config.market, "FLAT").values
    assert f["relative_volume"] == pytest.approx(1.0)                     # same volume, same time of day
    assert f["avg_daily_traded_value"] == pytest.approx(100 * 100 * 25)  # 25 bars/day, typical price 100
    assert f["gap_pct"] == pytest.approx(0.0) and f["momentum_pct"] == pytest.approx(0.0)
    assert f["atr_pct"] == pytest.approx(0.0)
    assert np.isnan(f["relative_strength_pct"])                          # no index supplied


def test_relative_volume_doubles_when_todays_volume_doubles(config):
    df = _flat_frame()
    last_day = df["timestamp"].dt.date.max()
    df.loc[df["timestamp"].dt.date == last_day, "volume"] = 200.0
    f = compute_symbol_features(complete_bars(df, "15m", AS_OF, config.market), None, "15m",
                                config.stock_selection, config.market).values
    assert f["relative_volume"] == pytest.approx(2.0)


# ---------------------------------------------------------- universe / duplicates ---
def test_universe_has_no_duplicate_symbols_and_merges_indices():
    u = load_universe(date(2026, 11, 6))
    assert len(u.symbols) == len(set(u.symbols)) == 58
    hdfc = next(m for m in u.members if m.symbol == "HDFCBANK")
    assert sorted(hdfc.indices) == ["BANKNIFTY", "NIFTY50"]
    assert any("POINT_IN_TIME" in w for w in u.warnings)


def test_dedupe_symbols_normalises_and_keeps_order():
    assert dedupe_symbols(["tcs", "TCS", "infy", " TCS "]) == ["TCS", "INFY"]


def test_point_in_time_membership():
    before = load_universe(date(2026, 8, 19)).symbols
    after = load_universe(date(2026, 11, 6)).symbols
    assert "WIPRO" in before and "BSE" not in before
    assert "BSE" in after and "WIPRO" not in after
    assert "HEROMOTOCO" in load_universe(date(2025, 9, 29)).symbols
    assert "HEROMOTOCO" not in after


def test_duplicate_input_keys_are_reported(engine):
    stocks, idx = synthetic_universe_market(names(12))
    stocks["stk00"] = stocks["STK00"].copy()  # same symbol, different case
    res = engine.run(stocks, idx, "15m", AS_OF, data_label=LABEL)
    assert any("DUPLICATE_SYMBOL_INPUT" in w for w in res.warnings)
    assert res.shortlist_symbols.count("STK00") <= 1


# --------------------------------------------------------------- filters / top N ---
def test_top_10_selection_from_larger_universe(engine):
    stocks, idx = synthetic_universe_market(names(15))
    res = engine.run(stocks, idx, "15m", AS_OF, make_universe(names(15)), LABEL)
    assert res.n_eligible == 15 and len(res.shortlist) == 10
    assert res.shortlist_symbols == list(res.ranking["symbol"].head(10))
    scores = [o.score for o in res.shortlist]
    assert scores == sorted(scores, reverse=True)


def test_top_n_is_configurable_and_fewer_eligible_returns_all(config):
    cfg = config.model_copy(deep=True)
    cfg.stock_selection.top_n = 5
    stocks, idx = synthetic_universe_market(names(12))
    assert len(StockSelectionEngine(cfg).run(stocks, idx, "15m", AS_OF, data_label=LABEL).shortlist) == 5
    few, idx4 = synthetic_universe_market(names(4))
    assert len(StockSelectionEngine(config).run(few, idx4, "15m", AS_OF, data_label=LABEL).shortlist) == 4


def test_liquidity_filter_excludes_illiquid_and_cheap_symbols(engine):
    syms = names(12)
    stocks, idx = synthetic_universe_market(syms, base_volume={"STK03": 4_000.0}, base_price={"STK05": 20.0})
    res = engine.run(stocks, idx, "15m", AS_OF, make_universe(syms), LABEL)
    excluded = {e.symbol: " ".join(e.reasons) for e in res.exclusions}
    assert "LOW_AVG_VOLUME" in excluded["STK03"] and "LOW_TRADED_VALUE" in excluded["STK03"]
    assert "LOW_PRICE" in excluded["STK05"]
    assert "STK03" not in res.shortlist_symbols and "STK05" not in res.ranking["symbol"].values


def test_invalid_data_symbol_is_excluded_not_silently_ranked(engine):
    syms = names(12)
    stocks, idx = synthetic_universe_market(syms)
    stocks["STK02"].loc[7, "high"] = stocks["STK02"].loc[7, "low"] - 1
    res = engine.run(stocks, idx, "15m", AS_OF, make_universe(syms), LABEL)
    reasons = {e.symbol: e.reasons for e in res.exclusions}
    assert any("INVALID_OHLC" in r for r in reasons["STK02"])
    assert "STK02" not in res.ranking["symbol"].values


def test_stale_symbol_missing_symbol_and_short_history(engine):
    syms = names(12) + ["NODATA"]
    stocks, idx = synthetic_universe_market(names(12))
    stocks["STK01"] = stocks["STK01"].iloc[:-25].reset_index(drop=True)          # last session missing -> stale
    short, _ = synthetic_universe_market(["STK07"], n_sessions=8)
    stocks["STK07"] = short["STK07"]                                              # only 8 sessions
    res = engine.run(stocks, idx, "15m", AS_OF, make_universe(syms), LABEL)
    reasons = {e.symbol: " ".join(e.reasons) for e in res.exclusions}
    assert "NO_DATA" in reasons["NODATA"]
    assert "STALE_DATA" in reasons["STK01"]
    assert "INSUFFICIENT_HISTORY" in reasons["STK07"] or "MISSING" in reasons["STK07"]


def test_missing_index_flags_but_still_ranks(engine):
    stocks, _ = synthetic_universe_market(names(12))
    res = engine.run(stocks, None, "15m", AS_OF, data_label=LABEL)
    assert any("NO_INDEX_DATA" in w for w in res.warnings)
    assert all("NO_INDEX_DATA" in o.risk_flags for o in res.shortlist)
    assert len(res.shortlist) == 10  # coverage 0.8 still meets the minimum


# ---------------------------------------------------------- look-ahead / determinism ---
def test_no_lookahead_future_bars_do_not_change_ranking(engine):
    syms = names(12)
    stocks, idx = synthetic_universe_market(syms)
    midday = datetime(2026, 11, 6, 12, 0, tzinfo=IST)
    base = engine.run(stocks, idx, "15m", midday, make_universe(syms), LABEL)

    tampered = {s: d.copy() for s, d in stocks.items()}
    for d in tampered.values():
        future = d["timestamp"] > pd.Timestamp(midday)
        d.loc[future, ["open", "high", "low", "close"]] *= 3.0     # absurd future prices
        d.loc[future, "volume"] *= 50
    again = engine.run(tampered, idx, "15m", midday, make_universe(syms), LABEL)
    pd.testing.assert_frame_equal(base.ranking, again.ranking)
    assert base.shortlist_symbols == again.shortlist_symbols


def test_results_are_deterministic(engine):
    stocks, idx = synthetic_universe_market(names(12))
    a = engine.run(stocks, idx, "15m", AS_OF, data_label=LABEL)
    b = engine.run(stocks, idx, "15m", AS_OF, data_label=LABEL)
    pd.testing.assert_frame_equal(a.ranking, b.ranking)


@pytest.mark.parametrize("tf", ["5m", "15m", "1h"])
def test_all_timeframes_supported(engine, tf):
    stocks, idx = synthetic_universe_market(names(12), tf=tf)
    res = engine.run(stocks, idx, tf, AS_OF, data_label=LABEL)
    assert len(res.shortlist) == 10
    assert all(o.timeframe.value == tf for o in res.shortlist)


# ------------------------------------------------------------- output contract ---
def test_shortlist_follows_common_output_contract(engine):
    stocks, idx = synthetic_universe_market(names(12))
    res = engine.run(stocks, idx, "15m", AS_OF, make_universe(names(12)), LABEL)
    for o in res.shortlist:
        assert isinstance(o, EngineOutput)
        assert o.engine_name == "stock_selection" and o.direction == Direction.NEUTRAL
        assert o.model_status == ModelStatus.RULE_BASED
        assert 0 <= o.score <= 1 and 0 <= o.confidence <= 1 and 0 <= o.uncertainty <= 1
        assert "SYNTHETIC_DATA" in o.risk_flags
        assert len(o.supporting_evidence) >= 10
        EngineOutput.model_validate(o.to_json_dict())  # survives a JSON round trip


def test_results_can_be_saved_and_reloaded(engine, tmp_path):
    stocks, idx = synthetic_universe_market(names(12))
    res = engine.run(stocks, idx, "15m", AS_OF, data_label=LABEL)
    paths = res.save(tmp_path)
    assert all(p.exists() for p in paths.values())
    loaded = json.loads(paths["shortlist"].read_text())
    assert [EngineOutput.model_validate(x).symbol for x in loaded] == res.shortlist_symbols
    assert "does not guarantee profit" in paths["summary"].read_text()


# ------------------------------------------------------------------------ adapter ---
def test_run_from_adapter_refuses_without_provider(engine):
    with pytest.raises(ProviderNotConfiguredError):
        engine.run_from_adapter(UnconfiguredAdapter(), make_universe(names(3)), "15m", AS_OF)


def test_run_from_synthetic_adapter(engine):
    syms = names(12)
    adapter = SyntheticMarketAdapter(syms, END_DATE, n_sessions=25)
    res = engine.run_from_adapter(adapter, make_universe(syms), "15m", AS_OF)
    assert len(res.shortlist) == 10 and "SYNTHETIC_DATA" in res.shortlist[0].risk_flags
