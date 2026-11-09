"""Tests for the common engine output contract."""
import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from shared.contracts import (
    DataQuality, DataQualityStatus, Direction, EngineOutput, Evidence, IST, ModelStatus, Timeframe, normalize_symbol,
)

REQUIRED_FIELDS = {
    "engine_name", "engine_version", "timestamp", "symbol", "timeframe", "prediction_horizon", "direction",
    "score", "confidence", "uncertainty", "supporting_evidence", "risk_flags", "data_quality", "model_status",
}


def make(**over):
    base = dict(
        engine_name="demo", engine_version="1.0.0", timestamp=datetime(2026, 11, 6, 15, 30, tzinfo=IST),
        symbol="RELIANCE", timeframe="15m", prediction_horizon="1d", direction="NEUTRAL", score=0.5,
        confidence=0.6, uncertainty=0.4, supporting_evidence=[Evidence(factor="x", value=1.0)], risk_flags=[],
        data_quality=DataQuality(status="GOOD", score=1.0, n_observations=100), model_status="RULE_BASED",
    )
    base.update(over)
    return EngineOutput(**base)


def test_contract_has_all_required_fields():
    assert REQUIRED_FIELDS <= set(EngineOutput.model_fields)


def test_valid_output_and_ist_conversion():
    out = make(timestamp=datetime.fromisoformat("2026-11-06T10:00:00+00:00"))
    assert out.timestamp.utcoffset().total_seconds() == 5.5 * 3600
    assert out.direction == Direction.NEUTRAL and out.timeframe == Timeframe.M15


def test_json_round_trip():
    out = make(risk_flags=["EXTREME_GAP"])
    again = EngineOutput.model_validate(json.loads(json.dumps(out.to_json_dict())))
    assert again == out


@pytest.mark.parametrize("bad", [
    dict(timestamp=datetime(2026, 11, 6, 15, 30)),          # naive datetime
    dict(score=1.5), dict(confidence=-0.1), dict(uncertainty=float("nan")),
    dict(symbol="TCS.NS"), dict(symbol="NSE:TCS"),
    dict(timeframe="2m"), dict(prediction_horizon="tomorrow"), dict(prediction_horizon="0d"),
    dict(direction="HOLD"), dict(model_status="MAGIC"),
    dict(risk_flags=["lowercase_flag"]), dict(engine_version="v1"),
    dict(unexpected_field=1),
])
def test_invalid_outputs_rejected(bad):
    with pytest.raises((ValidationError, TypeError, ValueError)):
        make(**bad)


def test_buy_requires_evidence():
    with pytest.raises(ValidationError):
        make(direction="BUY", supporting_evidence=[])


def test_buy_sell_rejected_on_poor_or_unknown_data():
    for status in ("POOR", "UNKNOWN"):
        with pytest.raises(ValidationError):
            make(direction="SELL", data_quality=DataQuality(status=status, score=0.2))


def test_buy_rejected_for_untrained_model_but_no_trade_allowed():
    with pytest.raises(ValidationError):
        make(direction="BUY", model_status="UNTRAINED")
    assert make(direction="NO_TRADE", model_status="UNTRAINED").direction == Direction.NO_TRADE


def test_valid_buy_allowed_with_evidence_good_data():
    assert make(direction="BUY", model_status="VALIDATED").direction == Direction.BUY


def test_evidence_rejects_nan():
    with pytest.raises(ValidationError):
        Evidence(factor="x", value=float("nan"))


def test_symbol_normalisation():
    assert normalize_symbol(" m&m ") == "M&M"
    assert normalize_symbol("bajaj-auto") == "BAJAJ-AUTO"
    assert normalize_symbol("nifty 50") == "NIFTY 50"
    with pytest.raises(ValueError):
        normalize_symbol("")
