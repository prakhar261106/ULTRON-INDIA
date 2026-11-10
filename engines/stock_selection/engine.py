"""ULTRON Stock Selection Engine (v1.0.0).

Picks the NSE stocks that are worth LOOKING AT for intraday analysis. It does not predict
direction and a high rank does NOT mean a trade will be profitable.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

from config.settings import UltronConfig, load_config
from config.universe import Universe, dedupe_symbols
from engines.stock_selection.features import (
    FEATURE_NAMES, SymbolFeatures, apply_sector_strength, compute_symbol_features)
from engines.stock_selection.ranking import feature_coverage, rank_candidates
from shared.contracts.engine_output import EngineOutput, Evidence
from shared.contracts.market_data import MarketDataAdapter, MarketDataError, ProviderNotConfiguredError
from shared.contracts.types import IST, INDEX_NIFTY50, Direction, ModelStatus, Timeframe, normalize_symbol
from shared.data_validation import ValidationReport, complete_bars, validate_candles
from shared.logging import get_logger, log_engine_event

ENGINE_NAME = "stock_selection"
ENGINE_VERSION = "1.0.0"

METHODOLOGY = """\
ULTRON Stock Selection - methodology (v1.0.0)
1. Universe: NIFTY 50 + NIFTY BANK constituents active on the as-of date, duplicates removed.
2. Look-ahead guard: only candles whose bar has CLOSED at as-of are used.
3. Data gate: each symbol's candles are validated; ERROR-level problems exclude the symbol.
4. Features: traded value, per-bar liquidity, relative volume, ATR%, volatility, gap, momentum,
   relative strength vs NIFTY 50, sector strength (see features.py for exact formulas).
5. Hard filters: minimum price, average daily volume, average daily traded value, history length,
   feature coverage. Failing symbols are excluded and the reason is reported.
6. Ranking: every feature becomes a percentile rank across eligible symbols (signed features are
   ranked by absolute size); the composite is the weighted average of percentiles. Weights are in
   config/ultron_config.yaml.
7. Shortlist: the top N (default 10) by composite score; ties broken alphabetically.
LIMITS: ranking measures activity and liquidity, NOT expected profit. No bid-ask spread or order-book
depth is used. Traded value is approximated from candles. A high rank guarantees nothing.
"""


@dataclass
class Exclusion:
    symbol: str
    reasons: List[str]


@dataclass
class StockSelectionResult:
    as_of: datetime
    timeframe: Timeframe
    data_label: str
    universe_size: int
    shortlist: List[EngineOutput]
    ranking: pd.DataFrame
    exclusions: List[Exclusion]
    warnings: List[str]
    methodology: str = METHODOLOGY
    weights: Dict[str, float] = field(default_factory=dict)

    @property
    def n_eligible(self) -> int:
        return len(self.ranking)

    @property
    def shortlist_symbols(self) -> List[str]:
        return [o.symbol for o in self.shortlist]

    def summary_dict(self) -> dict:
        return {
            "engine_name": ENGINE_NAME,
            "engine_version": ENGINE_VERSION,
            "as_of": self.as_of.isoformat(),
            "timeframe": self.timeframe.value,
            "data_label": self.data_label,
            "universe_size": self.universe_size,
            "n_eligible": self.n_eligible,
            "n_excluded": len(self.exclusions),
            "shortlist": self.shortlist_symbols,
            "warnings": self.warnings,
            "weights": self.weights,
            "disclaimer": "A high rank does not guarantee profit. Research/paper-trading use only.",
        }

    def save(self, directory: str | Path, prefix: str = "stock_selection") -> Dict[str, Path]:
        """Write ranking CSV + shortlist/exclusion/summary JSON files; returns their paths."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "ranking": d / f"{prefix}_ranking.csv",
            "shortlist": d / f"{prefix}_shortlist.json",
            "exclusions": d / f"{prefix}_exclusions.json",
            "summary": d / f"{prefix}_summary.json",
        }
        self.ranking.to_csv(paths["ranking"], index=False)
        paths["shortlist"].write_text(
            json.dumps([o.to_json_dict() for o in self.shortlist], indent=2, ensure_ascii=False), encoding="utf-8")
        paths["exclusions"].write_text(
            json.dumps([{"symbol": e.symbol, "reasons": e.reasons} for e in self.exclusions], indent=2), encoding="utf-8")
        paths["summary"].write_text(json.dumps(self.summary_dict(), indent=2, default=str), encoding="utf-8")
        return paths


class StockSelectionEngine:
    def __init__(self, config: UltronConfig | None = None, logger: logging.Logger | None = None):
        self.config = config or load_config()
        self.cfg = self.config.stock_selection
        self.log = logger or get_logger("ultron.stock_selection")

    # ------------------------------------------------------------------ data access ---
    def run_from_adapter(
        self,
        adapter: MarketDataAdapter,
        universe: Universe,
        timeframe: Timeframe | str,
        as_of: datetime,
        index_symbol: str = INDEX_NIFTY50,
        data_label: Optional[str] = None,
    ) -> StockSelectionResult:
        """Fetch candles through ``adapter`` and run. A missing provider raises - never fakes data."""
        tf = Timeframe.parse(timeframe)
        as_of = as_of.astimezone(IST)
        start = as_of - timedelta(days=int(self.cfg.lookback_sessions * 1.6) + 7)
        candles: Dict[str, pd.DataFrame] = {}
        fetch_errors: Dict[str, str] = {}
        for sym in dedupe_symbols(universe.symbols):
            try:
                candles[sym] = adapter.get_historical_candles(sym, tf, start, as_of)
            except ProviderNotConfiguredError:
                raise
            except MarketDataError as exc:
                fetch_errors[sym] = str(exc)
        try:
            index_df = adapter.get_historical_candles(index_symbol, tf, start, as_of)
        except ProviderNotConfiguredError:
            raise
        except MarketDataError:
            index_df = None
        result = self.run(candles, index_df, tf, as_of, universe, data_label or f"ADAPTER:{adapter.name}")
        for sym, msg in fetch_errors.items():
            result.exclusions.append(Exclusion(sym, [f"FETCH_ERROR: {msg}"]))
        if self.config.data_provider.require_adjusted_prices and adapter.adjusted_for_corporate_actions is not True:
            result.warnings.append(
                "PRICES_NOT_CONFIRMED_ADJUSTED: the adapter does not guarantee split/bonus-adjusted prices; "
                "gap and momentum may be distorted around corporate actions.")
        return result

    # ------------------------------------------------------------------------- core ---
    def run(
        self,
        candles_by_symbol: Mapping[str, pd.DataFrame],
        index_candles: Optional[pd.DataFrame],
        timeframe: Timeframe | str,
        as_of: datetime,
        universe: Optional[Universe] = None,
        data_label: str = "UNLABELLED",
    ) -> StockSelectionResult:
        tf = Timeframe.parse(timeframe)
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        as_of = as_of.astimezone(IST)
        cfg, market = self.cfg, self.config.market
        warnings: List[str] = []
        exclusions: List[Exclusion] = []

        # --- normalise input keys, drop duplicate symbols ------------------------------
        data: Dict[str, pd.DataFrame] = {}
        for key, frame in candles_by_symbol.items():
            sym = normalize_symbol(key)
            if sym in data:
                warnings.append(f"DUPLICATE_SYMBOL_INPUT: {sym} supplied more than once; first copy used.")
                continue
            data[sym] = frame

        if universe is not None:
            symbols = dedupe_symbols(universe.symbols)
            sector_of = universe.sector_of
            warnings.extend(universe.warnings)
        else:
            symbols = dedupe_symbols(data.keys())
            sector_of = {}

        # --- index data (for relative strength) ----------------------------------------
        index_ok: Optional[pd.DataFrame] = None
        if index_candles is None:
            warnings.append("NO_INDEX_DATA: no NIFTY 50 candles supplied; relative_strength_pct and sector_strength_pct unavailable.")
        else:
            idx_report = validate_candles(index_candles, tf, as_of, market, self.config.data_quality, symbol=INDEX_NIFTY50)
            if idx_report.is_valid:
                index_ok = complete_bars(index_candles, tf, as_of, market)
            else:
                warnings.append(f"NO_INDEX_DATA: NIFTY 50 candles failed validation {idx_report.codes}.")

        # --- per-symbol validation + features -----------------------------------------
        reports: Dict[str, ValidationReport] = {}
        feats: Dict[str, SymbolFeatures] = {}
        for sym in symbols:
            frame = data.get(sym)
            if frame is None:
                exclusions.append(Exclusion(sym, ["NO_DATA"]))
                continue
            rep = validate_candles(frame, tf, as_of, market, self.config.data_quality, symbol=sym)
            reports[sym] = rep
            if not rep.is_valid:
                exclusions.append(Exclusion(sym, [f"DATA_INVALID:{e.code}" for e in rep.errors]))
                continue
            sf = compute_symbol_features(complete_bars(frame, tf, as_of, market), index_ok, tf, cfg, market, sym)
            if sf.n_prior_sessions < cfg.min_sessions:
                exclusions.append(Exclusion(sym, [f"INSUFFICIENT_HISTORY: {sf.n_prior_sessions} prior sessions < {cfg.min_sessions}"]))
                continue
            feats[sym] = sf

        # --- sector strength (needs all symbols' relative strength; records reasons if unavailable)
        apply_sector_strength(feats, sector_of)

        # --- hard filters ---------------------------------------------------------------
        passed: Dict[str, SymbolFeatures] = {}
        for sym, f in feats.items():
            reasons = []
            if f.last_close < cfg.min_price_inr:
                reasons.append(f"LOW_PRICE: {f.last_close:.2f} < {cfg.min_price_inr}")
            if not (f.avg_daily_volume >= cfg.min_avg_daily_volume):
                reasons.append(f"LOW_AVG_VOLUME: {f.avg_daily_volume:,.0f} < {cfg.min_avg_daily_volume:,.0f}")
            tv = f.values["avg_daily_traded_value"]
            if not (tv >= cfg.min_avg_daily_traded_value_inr):
                reasons.append(f"LOW_TRADED_VALUE: {tv:,.0f} < {cfg.min_avg_daily_traded_value_inr:,.0f}")
            if reasons:
                exclusions.append(Exclusion(sym, reasons))
            else:
                passed[sym] = f

        weights = self.cfg.normalised_weights()
        table = pd.DataFrame({s: f.values for s, f in passed.items()}).T.reindex(columns=FEATURE_NAMES).astype(float)
        table.index.name = "symbol"
        if not table.empty:
            cov = feature_coverage(table, weights)
            too_thin = cov.index[cov < cfg.min_feature_coverage - 1e-9]  # tolerance: 0.7999999 must count as 0.8
            for sym in too_thin:
                gone = ", ".join(f"{k}({v})" for k, v in passed[sym].missing.items())
                exclusions.append(Exclusion(sym, [f"INSUFFICIENT_FEATURES: coverage {cov[sym]:.2f} < {cfg.min_feature_coverage}; missing: {gone}"]))
                passed.pop(sym)
            table = table.drop(index=list(too_thin))

        if table.empty:
            ranking = pd.DataFrame(columns=["symbol", *FEATURE_NAMES, "composite_score", "feature_coverage", "rank", "sector", "missing_features"])
            warnings.append("NO_ELIGIBLE_SYMBOLS: nothing passed the data and liquidity filters.")
        else:
            ranking = rank_candidates(table, weights)
            ranking["sector"] = ranking["symbol"].map(lambda s: sector_of.get(s, "UNKNOWN"))
            ranking["missing_features"] = ranking["symbol"].map(
                lambda s: "; ".join(f"{k}:{v}" for k, v in passed[s].missing.items()))
            n_missing = int((ranking["missing_features"] != "").sum())
            if n_missing:
                counts: Dict[str, int] = {}
                for s_ in ranking["symbol"]:
                    for k in passed[s_].missing:
                        counts[k] = counts.get(k, 0) + 1
                warnings.append(f"MISSING_FEATURES: {n_missing} ranked symbols lack at least one feature "
                                f"{counts}; unavailable features are left out of the composite (re-normalised weights), never scored as 0.")
        if 0 < len(ranking) < 5:
            warnings.append(f"SMALL_RANKING_UNIVERSE: only {len(ranking)} eligible symbols; percentile ranks are coarse.")
        warnings.append("NO_SPREAD_OR_DEPTH_DATA: liquidity is approximated from candles; bid-ask spread and market depth are not used.")

        # --- build contract outputs for the shortlist -----------------------------------
        shortlist: List[EngineOutput] = []
        for _, row in ranking.head(cfg.top_n).iterrows():
            sym = row["symbol"]
            shortlist.append(self._to_output(sym, row, passed[sym], reports[sym], tf, as_of, len(ranking), weights,
                                             sector_of.get(sym, "UNKNOWN"), data_label, universe, index_ok is not None))

        log_engine_event(self.log, ENGINE_NAME, "run_complete", timeframe=tf.value, as_of=as_of.isoformat(),
                         universe=len(symbols), eligible=len(ranking), excluded=len(exclusions), shortlisted=len(shortlist))
        return StockSelectionResult(as_of, tf, data_label, len(symbols), shortlist, ranking, exclusions, warnings,
                                    METHODOLOGY, weights)

    # ------------------------------------------------------------------------ output ---
    def _to_output(self, sym, row, sf, report, tf, as_of, n_eligible, weights, sector, data_label, universe, index_ok) -> EngineOutput:
        cfg = self.cfg
        dq = report.to_data_quality()
        coverage = float(row["feature_coverage"])
        confidence = float(min(1.0, max(0.0, dq.score * coverage)))

        evidence = [
            Evidence(factor="rank", value=float(row["rank"]), detail=f"rank {int(row['rank'])} of {n_eligible} eligible symbols"),
            Evidence(factor="composite_score", value=float(row["composite_score"]), detail="weighted average of feature percentile ranks (0-1)"),
            Evidence(factor="sector", value=sector, detail="project-defined sector grouping"),
        ]
        for name in FEATURE_NAMES:
            val = row[name]
            if isinstance(val, float) and math.isnan(val):
                why = sf.missing.get(name, "UNAVAILABLE")
                evidence.append(Evidence(factor=name, value=None,
                    detail=f"NOT AVAILABLE ({why}); left out of this symbol's composite and weights re-normalised - not treated as zero"))
                continue
            pct = row.get(f"pct_{name}")
            evidence.append(Evidence(
                factor=name, value=round(float(val), 6),
                detail=f"percentile {float(pct):.2f} of {n_eligible}; weight {weights.get(name, 0):.2f}"))

        flags: List[str] = []
        gap = row["gap_pct"]
        if not math.isnan(gap) and abs(gap) >= cfg.gap_flag_pct:
            flags.append("EXTREME_GAP")
        rv = row["relative_volume"]
        if not math.isnan(rv) and rv < cfg.low_relative_volume_flag:
            flags.append("LOW_RELATIVE_VOLUME")
        vp = row.get("pct_volatility_pct")
        if vp is not None and not math.isnan(vp) and vp >= cfg.high_volatility_percentile_flag:
            flags.append("HIGH_RELATIVE_VOLATILITY")
        if dq.status.value == "DEGRADED":
            flags.append("DATA_DEGRADED")
        if sf.missing:
            flags.append("MISSING_FEATURES")
        if not index_ok:
            flags.append("NO_INDEX_DATA")
        if sf.current_session != as_of.date():
            flags.append("FEATURES_FROM_PRIOR_SESSION")
        if universe is not None and not universe.membership_history_verified:
            flags.append("UNIVERSE_NOT_POINT_IN_TIME")
        if "SYNTHETIC" in data_label.upper():
            flags.append("SYNTHETIC_DATA")

        return EngineOutput(
            engine_name=ENGINE_NAME, engine_version=ENGINE_VERSION, timestamp=as_of, symbol=sym, timeframe=tf,
            prediction_horizon=cfg.prediction_horizon, direction=Direction.NEUTRAL,
            score=float(row["composite_score"]), confidence=confidence, uncertainty=1.0 - confidence,
            supporting_evidence=evidence, risk_flags=sorted(set(flags)), data_quality=dq,
            model_status=ModelStatus.RULE_BASED,
            metadata={
                "rank": int(row["rank"]), "n_eligible": n_eligible, "data_label": data_label,
                "missing_features": dict(sf.missing), "feature_coverage": round(coverage, 4),
                "current_session": str(sf.current_session), "weights": weights,
                "note": "Selection only: NEUTRAL direction. confidence = data_quality_score x feature_coverage; "
                        "uncertainty = 1 - confidence. A high rank does not guarantee profit.",
            },
        )
