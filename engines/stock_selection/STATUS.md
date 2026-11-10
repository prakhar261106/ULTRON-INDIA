# Stock Selection Engine — Status Report (Day 1)

Engine version 1.0.0 · Contract version 1.0.0 · Report date: 2026-11-07

## Completed
- Shared output contract (`EngineOutput`) with validation and safety rules.
- Configuration system (`config/ultron_config.yaml`, validated, typo-proof) and universe loader (dedupe, membership dates).
- Data validation (13+ checks) and NSE session helpers; logging with secret redaction.
- Provider-independent `MarketDataAdapter` interface, `UnconfiguredAdapter`, `CsvMarketDataAdapter`.
- Stock Selection Engine: features, percentile ranking, filters, top-10 shortlist, JSON/CSV export.
- 96 automated tests, all passing (pandas 2.3.3 and pandas 3.0.6, Python 3.12); Colab notebook executed end-to-end in a sandbox.

## Incomplete / deliberately not done
- No live provider (Dhan, Zerodha, ...) — not selected yet; no live-data code exists.
- Other engines (news, technical, deep learning, correlation, fusion, master, risk) — not started, by design.

## Blocked
- Running on real NSE data is blocked until the user chooses a data provider or supplies exported candle files.

## Unverified
- Behaviour on real market data (all tests/demos use SYNTHETIC data).
- Not yet run inside the user's own Google Colab session.
- NIFTY 50 list checked against web sources dated up to 30 Sep 2026; NIFTY BANK has 13 of 14 constituents confirmed (14th unknown).
- Sector groupings are project-defined, not official NSE classifications.
- NSE holiday calendar is not built in (`market.holidays` is empty).
- Whether ranking has any predictive value — this has NOT been tested and no such claim is made.

## Known limitations
- No spread / order-book depth; traded value approximated from candles.
- Membership history incomplete (survivorship-bias risk for past dates; flagged as `UNIVERSE_NOT_POINT_IN_TIME`).
- Special sessions (e.g. Diwali Muhurat) are reported as `OFF_SESSION_BARS` warnings, not handled as normal sessions.
- Prices must be split/bonus-adjusted by the provider; the engine cannot verify this.

## Bug found and fixed during testing
- Feature-coverage threshold failed on floating-point rounding (0.7999… < 0.8), wrongly excluding every symbol when index data was missing. Fixed with a tolerance; regression covered by `test_missing_index_flags_but_still_ranks`.
- `sector_strength_pct` showed NaN in notebook Cell 8: the standalone feature function cannot compute it (needs sector peers; the engine always did). Fixed by adding `apply_sector_strength()` (used by both engine and notebook), recording a reason for every unavailable feature, adding the `MISSING_FEATURES` risk flag, and making the ranking explicitly never score a missing value as 0. Covered by `tests/test_missing_features.py` (10 tests; a mutation check confirmed they fail if NaN were turned into 0).
