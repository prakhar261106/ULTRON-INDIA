# ULTRON Stock Selection Engine (v1.0.0)

**Purpose:** from the NIFTY 50 + BANK NIFTY universe, shortlist the stocks that are most *worth analysing*
for intraday work today: liquid, active, and moving relative to the market.

**It does not predict direction, and a high rank does not guarantee profit.** Its output direction is always
`NEUTRAL`. Later engines (technical, news, deep learning, ...) decide whether there is a trade.

## How the ranking works

1. **Universe** - NIFTY 50 + NIFTY BANK members active on the as-of date; duplicates (e.g. HDFCBANK is in both) are removed.
2. **Look-ahead guard** - only candles whose bar has *closed* by the as-of time are used.
3. **Data gate** - candles are validated; ERROR-level problems (duplicates, invalid OHLC, missing values, stale data, too many missing candles, ...) exclude the symbol.
4. **Features** (formulas in `features.py`):

| Feature | Meaning |
|---|---|
| `avg_daily_traded_value` | average of daily sum(typical price x volume) over prior sessions (INR, approximate) |
| `liquidity_bar_value` | median traded value per bar over prior sessions |
| `relative_volume` | today's volume so far / average volume at the same time of day in prior sessions |
| `atr_pct` | ATR(14) as % of price, true range computed inside each session |
| `volatility_pct` | within-session return volatility, scaled to a daily equivalent (%) |
| `gap_pct` | latest session open vs previous close (%), signed |
| `momentum_pct` | % change over the last 20 bars, signed |
| `relative_strength_pct` | stock momentum minus NIFTY 50 momentum over the same timestamps, signed |
| `sector_strength_pct` | mean relative strength of the stock's sector members, signed |

5. **Hard filters** (all in `config/ultron_config.yaml`): minimum price, average daily volume, average daily traded value, at least 10 prior sessions, at least 80% of ranking weight computable.
6. **Ranking** - each feature becomes a *percentile rank* across eligible symbols (signed features by absolute size, because the engine looks for movement either way); composite = weighted average of percentiles; weights are configurable.
7. **Shortlist** - top N (default 10); ties broken alphabetically so results are reproducible.

## Using it

```python
from datetime import datetime
from config.universe import load_universe
from engines.stock_selection import StockSelectionEngine
from shared.contracts import IST

engine = StockSelectionEngine()
universe = load_universe()                      # NIFTY 50 + BANK NIFTY, duplicates removed
result = engine.run_from_adapter(my_adapter, universe, "15m", datetime.now(IST))   # needs a real adapter
for out in result.shortlist:                    # list of EngineOutput (common contract)
    print(out.symbol, out.score, out.risk_flags)
result.save("data/processed/stock_selection")
```

Until a data provider adapter exists, use `CsvMarketDataAdapter` with your own exported candle files, or call
`engine.run(candles_by_symbol, index_candles, timeframe, as_of, universe)` directly.

## Output (common contract)

Each shortlisted stock is an `EngineOutput`: `direction=NEUTRAL`, `score` = composite percentile score (0-1),
`confidence` = data-quality score x feature coverage (a measure of how well the inputs were measured, **not** a
probability of profit), `uncertainty` = 1 - confidence, `supporting_evidence` = every feature value, its
percentile and weight, `risk_flags`, `data_quality`, `model_status=RULE_BASED`.

Risk flags: `EXTREME_GAP`, `LOW_RELATIVE_VOLUME`, `HIGH_RELATIVE_VOLATILITY`, `DATA_DEGRADED`, `NO_INDEX_DATA`,
`FEATURES_FROM_PRIOR_SESSION` (run before the open), `UNIVERSE_NOT_POINT_IN_TIME`, `SYNTHETIC_DATA`.

## Known limits

* No bid-ask spread or order-book depth: liquidity is approximated from candles.
* Traded value uses (H+L+C)/3 x volume, not exchange-reported turnover.
* Sector groupings are project-defined; sector strength uses only the universe members of each sector.
* Index (NIFTY 50) candles are needed for relative/sector strength; without them coverage drops to 80%.
* Historical membership is incomplete, so backtests over past dates carry survivorship-bias risk.
* The NSE holiday list is not built in; fill `market.holidays` in the config.
* Not yet run on real market data (no provider selected). Sample files are SYNTHETIC.

Samples: `samples/sample_input_SYNTHETIC_candles_1h.csv`, `samples/sample_output_SYNTHETIC.json`.
Tests: `tests/test_stock_selection.py`. Status: `STATUS.md`.
