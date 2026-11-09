# ULTRON-INDIA

Modular, AI-assisted **Indian stock-market (NSE) intraday intelligence and decision-support system**.

ULTRON analyses NSE stocks and, step by step, will produce **BUY / SELL / NO TRADE** signals with evidence,
confidence, uncertainty and an independently validated risk plan.

> **Research, backtesting and paper trading only.** ULTRON never places broker orders; any real trade is
> made manually by you. Nothing here is investment advice and **no profit is guaranteed**. A model or
> ranking is not "profitable" until it passes out-of-sample, walk-forward, cost-adjusted tests and
> subsequent paper trading.

## Build plan

| Day | Scope | Status |
|-----|-------|--------|
| 1 | Foundation, shared contract, config, validation, logging, **Stock Selection Engine** | Code complete, tests pass on synthetic data (see `engines/stock_selection/STATUS.md`) |
| 2 | News Intelligence + Technical Analysis | Not started |
| 3 | Deep Learning + Market Correlation | Not started |
| 4 | Pair A / Pair B fusion + Master Decision Engine | Not started |
| 5 | Independent Risk Engine + integration + paper trading | Not started |

## Repository layout

```
config/            ultron_config.yaml (all settings), universe.yaml + universe.py (NIFTY 50 + BANK NIFTY)
shared/contracts/  engine output schema (EngineOutput), market-data interface (MarketDataAdapter)
shared/data_validation/  candle validation + NSE session calendar helpers
shared/logging/    logger with secret redaction
engines/           stock_selection (Day 1); other engines arrive on later days
fusion/ risk/ backtesting/ paper_trading/ evaluation/ model_registry/   placeholders for later days
data/raw|processed|features/   local data (contents are git-ignored, only .gitkeep is tracked)
notebooks/         Google Colab notebooks
tests/             automated tests (pytest)
documentation/     guides
```

## Quick start (Google Colab)

1. Open `notebooks/day1_foundation_stock_selection.ipynb` in Colab (see `documentation/DAY1_UPLOAD_GUIDE.md`).
2. Run the cells from top to bottom.

Or in any Python 3.10+ environment:

```bash
pip install -r requirements.txt
python -m pytest
```

## Important rules

* **No fake live data.** The market-data provider is **not selected yet**. `UnconfiguredAdapter` refuses
  to return data. Synthetic data exists only in `tests/synthetic_data.py` and is always labelled SYNTHETIC.
* **No look-ahead.** Only candles whose bar has closed at the "as of" time are used.
* **Chronological evaluation only** (no shuffled time-series splits) once models arrive.
* **Secrets** live in environment variables / Colab Secrets, never in code, notebooks, logs or GitHub.
* All timestamps are **Asia/Kolkata**. Candle timestamps are bar **open** times.
* Universe lists must be verified against official NSE Indices files (`config/universe.yaml` explains what is verified).
