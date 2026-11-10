"""Tests for unavailable (NaN) features, e.g. sector_strength_pct.

Rule under test: a feature that cannot be computed is NaN with a recorded reason, is flagged
(MISSING_FEATURES), and is NEVER silently replaced by zero. SYNTHETIC data only.
"""
import math

import numpy as np
import pandas as pd
import pytest

from config.universe import Universe, UniverseMember
from engines.stock_selection import StockSelectionEngine
from engines.stock_selection.features import apply_sector_strength, compute_symbol_features
from engines.stock_selection.ranking import rank_candidates
from shared.data_validation import complete_bars
from tests.conftest import AS_OF, END_DATE, synthetic_universe_market

LABEL = "SYNTHETIC_TEST_DATA"
SECTORS = {"AAA": "Oil & Gas", "BBB": "Oil & Gas", "CCC": "Oil & Gas", "DDD": "IT", "EEE": "UNKNOWN"}


def feats_for(symbols, config, index_df, stocks):
    return {s: compute_symbol_features(complete_bars(stocks[s], "15m", AS_OF, config.market), index_df, "15m",
                                       config.stock_selection, config.market, s) for s in symbols}


@pytest.fixture(scope="module")
def market():
    return synthetic_universe_market(list(SECTORS))


def test_standalone_feature_call_reports_why_sector_strength_is_nan(config, market):
    """The root cause of the DEMO-RELIANCE NaN: sector strength needs universe context."""
    stocks, idx = market
    f = feats_for(["AAA"], config, idx, stocks)["AAA"]
    assert math.isnan(f.values["sector_strength_pct"])                      # NaN, not 0
    assert f.missing["sector_strength_pct"] == "PENDING_UNIVERSE_CONTEXT"   # with a reason
    assert not math.isnan(f.values["relative_strength_pct"])                # other features unaffected
    assert "relative_strength_pct" not in f.missing


def test_apply_sector_strength_is_mean_relative_strength_of_sector_members(config, market):
    stocks, idx = market
    feats = feats_for(list(SECTORS), config, idx, stocks)
    apply_sector_strength(feats, SECTORS)
    oil = np.mean([feats[s].values["relative_strength_pct"] for s in ("AAA", "BBB", "CCC")])
    for s in ("AAA", "BBB", "CCC"):
        assert feats[s].values["sector_strength_pct"] == pytest.approx(oil)
        assert "sector_strength_pct" not in feats[s].missing
    # single-member sector: equals its own relative strength
    assert feats["DDD"].values["sector_strength_pct"] == pytest.approx(feats["DDD"].values["relative_strength_pct"])


def test_unknown_sector_stays_nan_with_reason(config, market):
    stocks, idx = market
    feats = feats_for(["EEE"], config, idx, stocks)
    apply_sector_strength(feats, SECTORS)
    assert math.isnan(feats["EEE"].values["sector_strength_pct"])
    assert feats["EEE"].missing["sector_strength_pct"] == "SECTOR_UNKNOWN"


def test_no_index_data_means_no_sector_strength_not_zero(config, market):
    stocks, _ = market
    feats = feats_for(["AAA", "BBB"], config, None, stocks)
    apply_sector_strength(feats, SECTORS)
    for f in feats.values():
        assert math.isnan(f.values["relative_strength_pct"]) and f.missing["relative_strength_pct"] == "NO_INDEX_DATA"
        assert math.isnan(f.values["sector_strength_pct"]) and f.missing["sector_strength_pct"] == "NO_SECTOR_RELATIVE_STRENGTH"


# ---------------------------------------------------------------- ranking safety ---
def test_ranking_never_scores_missing_as_zero():
    f = pd.DataFrame({"relative_volume": [3.0, 2.0, 1.0], "sector_strength_pct": [1.0, np.nan, 2.0]}, index=["A", "B", "C"])
    f.index.name = "symbol"
    out = rank_candidates(f, {"relative_volume": 0.5, "sector_strength_pct": 0.5}).set_index("symbol")
    # B has only relative_volume (pct 2/3). If NaN were scored as 0 it would be 1/3.
    assert out.loc["B", "composite_score"] == pytest.approx(2 / 3)
    assert out.loc["B", "feature_coverage"] == pytest.approx(0.5)
    assert np.isnan(out.loc["B", "pct_sector_strength_pct"])


def test_row_with_no_available_features_gets_nan_and_ranks_last():
    f = pd.DataFrame({"relative_volume": [1.0, np.nan, 2.0], "atr_pct": [1.0, np.nan, 2.0]}, index=["A", "B", "C"])
    f.index.name = "symbol"
    out = rank_candidates(f, {"relative_volume": 0.5, "atr_pct": 0.5})
    b = out[out["symbol"] == "B"].iloc[0]
    assert np.isnan(b["composite_score"]) and b["feature_coverage"] == 0.0
    assert out["symbol"].iloc[-1] == "B"


# ---------------------------------------------------------------- engine behaviour ---
def test_engine_flags_missing_features_when_index_is_absent(config):
    stocks, _ = synthetic_universe_market([f"STK{i:02d}" for i in range(12)])
    res = StockSelectionEngine(config).run(stocks, None, "15m", AS_OF, data_label=LABEL)
    assert any(w.startswith("MISSING_FEATURES") for w in res.warnings)
    assert res.ranking["relative_strength_pct"].isna().all() and res.ranking["sector_strength_pct"].isna().all()
    assert (res.ranking["missing_features"] != "").all()
    for o in res.shortlist:
        assert "MISSING_FEATURES" in o.risk_flags
        assert o.metadata["missing_features"]["relative_strength_pct"] == "NO_INDEX_DATA"
        ev = {e.factor: e for e in o.supporting_evidence}
        assert ev["sector_strength_pct"].value is None                        # not 0
        assert "not treated as zero" in ev["sector_strength_pct"].detail
        assert o.confidence < 1.0                                             # coverage lowers confidence


def test_engine_has_no_missing_flag_when_everything_is_available(config):
    syms = [f"STK{i:02d}" for i in range(12)]
    stocks, idx = synthetic_universe_market(syms)
    uni = Universe(as_of=END_DATE, membership_history_verified=True,
                   members=[UniverseMember(symbol=s, sector="S", indices=["NIFTY50"]) for s in syms])
    res = StockSelectionEngine(config).run(stocks, idx, "15m", AS_OF, uni, LABEL)
    assert (res.ranking["missing_features"] == "").all()
    assert all("MISSING_FEATURES" not in o.risk_flags and o.metadata["missing_features"] == {} for o in res.shortlist)
    assert not any(w.startswith("MISSING_FEATURES") for w in res.warnings)


def test_engine_sector_value_matches_helper(config, market):
    """Engine result for a stock equals the helper's value, so notebook and engine agree."""
    stocks, idx = market
    uni = Universe(as_of=END_DATE, membership_history_verified=True,
                   members=[UniverseMember(symbol=s, sector=SECTORS[s], indices=["NIFTY50"]) for s in SECTORS])
    res = StockSelectionEngine(config).run(stocks, idx, "15m", AS_OF, uni, LABEL)
    feats = feats_for(list(SECTORS), config, idx, stocks)
    apply_sector_strength(feats, SECTORS)
    row = res.ranking.set_index("symbol")
    for s in ("AAA", "DDD"):
        assert row.loc[s, "sector_strength_pct"] == pytest.approx(feats[s].values["sector_strength_pct"])
    assert math.isnan(row.loc["EEE", "sector_strength_pct"]) and "SECTOR_UNKNOWN" in row.loc["EEE", "missing_features"]


def test_too_many_missing_features_excludes_symbol_with_named_reasons(config):
    cfg = config.model_copy(deep=True)
    cfg.stock_selection.momentum_lookback_bars = 5000      # impossible: momentum, RS and sector all unavailable
    stocks, idx = synthetic_universe_market([f"STK{i:02d}" for i in range(6)])
    res = StockSelectionEngine(cfg).run(stocks, idx, "15m", AS_OF, data_label=LABEL)
    assert res.shortlist == [] and res.n_eligible == 0 and "missing_features" in res.ranking.columns
    reasons = " ".join(r for e in res.exclusions for r in e.reasons)
    assert "INSUFFICIENT_FEATURES" in reasons and "momentum_pct(INSUFFICIENT_BARS)" in reasons
    assert any("NO_ELIGIBLE_SYMBOLS" in w for w in res.warnings)
