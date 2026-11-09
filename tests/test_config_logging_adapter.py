"""Tests for configuration, secrets, logging and the market-data adapter interface."""
from datetime import datetime, timedelta

import pytest
import yaml
from pydantic import ValidationError

from config.settings import DEFAULT_CONFIG_PATH, get_secret, load_config
from shared.contracts import (
    CsvMarketDataAdapter, IST, LiveQuote, MarketDataAdapter, MarketDataError, ProviderNotConfiguredError,
    UnconfiguredAdapter, is_quote_live,
)
from shared.logging import get_logger, redact_secrets


# ------------------------------------------------------------------ configuration ---
def test_default_config_loads(config):
    assert config.stock_selection.top_n == 10
    assert [t.value for t in config.timeframes] == ["5m", "15m", "1h"]
    assert config.data_provider.name is None  # provider deliberately not selected
    assert config.market.session_minutes == 375
    assert abs(sum(config.stock_selection.normalised_weights().values()) - 1.0) < 1e-9


def _write_cfg(tmp_path, mutate):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    mutate(raw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("mutate", [
    lambda r: r["stock_selection"]["weights"].update(not_a_feature=1.0),
    lambda r: r["stock_selection"]["weights"].update(relative_volume=-1.0),
    lambda r: r["stock_selection"].update(weights={k: 0 for k in r["stock_selection"]["weights"]}),
    lambda r: r["stock_selection"].update(min_sessions=99),
    lambda r: r["stock_selection"].update(top_nn=5),          # typo -> rejected, not ignored
    lambda r: r["market"].update(timezone="UTC"),
    lambda r: r["timeframes"].append("2m"),
])
def test_invalid_config_rejected(tmp_path, mutate):
    with pytest.raises((ValidationError, ValueError)):
        load_config(_write_cfg(tmp_path, mutate))


def test_get_secret_from_environment(monkeypatch):
    monkeypatch.setenv("ULTRON_TEST_SECRET", "abc")
    assert get_secret("ULTRON_TEST_SECRET") == "abc"
    assert get_secret("ULTRON_DOES_NOT_EXIST") is None
    with pytest.raises(RuntimeError):
        get_secret("ULTRON_DOES_NOT_EXIST", required=True)


# ---------------------------------------------------------------------- logging ---
def test_redact_secrets():
    text = redact_secrets("login api_key=ABC123 token: xyz789 password='hunter2' ok=1")
    assert "ABC123" not in text and "xyz789" not in text and "hunter2" not in text
    assert "ok=1" in text


def test_logger_redacts_and_does_not_duplicate_handlers(tmp_path):
    log = get_logger("ultron.test_logger", log_dir=tmp_path)
    n = len(log.handlers)
    log = get_logger("ultron.test_logger", log_dir=tmp_path)
    assert len(log.handlers) == n
    log.info("connecting with password=hunter2")
    for h in log.handlers:
        h.flush()
    content = (tmp_path / "ultron.test_logger.log").read_text()
    assert "hunter2" not in content and "REDACTED" in content


# ---------------------------------------------------------------- data adapter ---
def test_unconfigured_adapter_never_fakes_data():
    a = UnconfiguredAdapter()
    with pytest.raises(ProviderNotConfiguredError):
        a.get_historical_candles("TCS", "15m", datetime(2026, 1, 1, tzinfo=IST), datetime(2026, 1, 2, tzinfo=IST))
    with pytest.raises(ProviderNotConfiguredError):
        a.get_live_quotes(["TCS"])


def test_adapter_interface_is_abstract():
    with pytest.raises(TypeError):
        MarketDataAdapter()  # cannot be instantiated without implementing both methods


def _csv(path, rows, header="timestamp,open,high,low,close,volume"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_csv_adapter_reads_offset_timestamps(tmp_path):
    _csv(tmp_path / "TCS_15m.csv", ["2026-11-06 09:15:00+05:30,100,101,99,100.5,1000",
                                    "2026-11-06 09:30:00+05:30,100.5,102,100,101,1200"])
    df = CsvMarketDataAdapter(tmp_path).get_historical_candles(
        "tcs", "15m", datetime(2026, 11, 6, tzinfo=IST), datetime(2026, 11, 7, tzinfo=IST))
    assert len(df) == 2 and str(df["timestamp"].dt.tz) == "Asia/Kolkata"


def test_csv_adapter_rejects_naive_timestamps_unless_tz_declared(tmp_path):
    _csv(tmp_path / "TCS_15m.csv", ["2026-11-06 09:15:00,100,101,99,100.5,1000"])
    args = ("TCS", "15m", datetime(2026, 11, 6, tzinfo=IST), datetime(2026, 11, 7, tzinfo=IST))
    with pytest.raises(MarketDataError):
        CsvMarketDataAdapter(tmp_path).get_historical_candles(*args)
    ok = CsvMarketDataAdapter(tmp_path, naive_timestamps_tz="Asia/Kolkata").get_historical_candles(*args)
    assert len(ok) == 1


def test_csv_adapter_errors(tmp_path):
    a = CsvMarketDataAdapter(tmp_path)
    s, e = datetime(2026, 11, 6, tzinfo=IST), datetime(2026, 11, 7, tzinfo=IST)
    with pytest.raises(MarketDataError):
        a.get_historical_candles("NOFILE", "15m", s, e)
    _csv(tmp_path / "BAD_15m.csv", ["2026-11-06 09:15:00+05:30,1,2"], header="timestamp,open,high")
    with pytest.raises(MarketDataError):
        a.get_historical_candles("BAD", "15m", s, e)
    with pytest.raises(MarketDataError):
        a.get_live_quotes(["TCS"])


def test_live_quote_freshness_rules():
    now = datetime(2026, 11, 6, 11, 0, 0, tzinfo=IST)
    def q(delay, age=1.0):
        return LiveQuote(symbol="TCS", exchange_timestamp=now - timedelta(seconds=age), received_at=now,
                         last_price=100.0, data_delay_seconds=delay)
    assert is_quote_live(q(0.5), now)
    assert not is_quote_live(q(None), now)         # unknown delay is NOT live
    assert not is_quote_live(q(900.0), now)        # 15-minute delayed feed is NOT live
    assert not is_quote_live(q(0.5, age=60), now)  # old quote is NOT live
    with pytest.raises(ValidationError):
        LiveQuote(symbol="TCS", exchange_timestamp=datetime(2026, 11, 6, 11, 0), received_at=now, last_price=1.0)
