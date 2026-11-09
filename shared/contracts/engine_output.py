"""The common output contract that EVERY ULTRON engine must return.

Field guide (all fields are required unless a default is shown)
----------------------------------------------------------------
engine_name         str       Short engine id, e.g. ``"stock_selection"``.
engine_version      str       Semantic version of the engine code, e.g. ``"1.0.0"``.
timestamp           datetime  Time the output refers to ("as of"). Must be timezone-aware;
                              stored in Asia/Kolkata.
symbol              str       Plain NSE symbol (``"RELIANCE"``) or index name (``"NIFTY 50"``).
timeframe           Timeframe ``5m``, ``15m`` or ``1h``.
prediction_horizon  str       How far ahead the output applies: ``<number><m|h|d>``,
                              e.g. ``"30m"``, ``"4h"``, ``"1d"`` (``"1d"`` = same trading session).
direction           Direction BUY, SELL, NEUTRAL or NO_TRADE. Engines that do not take a side
                              (e.g. stock selection) must use NEUTRAL.
score               float     0..1. Engine-specific strength of the finding. NOT a probability.
confidence          float     0..1. The engine's own reliability estimate. NOT the probability
                              of profit unless ``model_status`` is CALIBRATED.
uncertainty         float     0..1. Higher means less certain. May be coupled with confidence
                              in simple rule-based engines; documented per engine.
supporting_evidence list      ``Evidence`` items that justify the output (factor, value, detail).
risk_flags          list[str] UPPER_SNAKE_CASE warnings, e.g. ``"EXTREME_GAP"``.
data_quality        object    ``DataQuality``: status, score, number of observations, issues.
model_status        enum      RULE_BASED, UNTRAINED, TRAINED_UNVALIDATED, VALIDATED,
                              CALIBRATED or DEGRADED.

Extra (optional) fields
-----------------------
contract_version    str       Version of THIS schema, so future changes stay traceable.
metadata            dict      Free-form engine-specific details. Other engines must not depend
                              on its contents.

Safety rules enforced by the validator
--------------------------------------
* BUY / SELL must carry at least one piece of supporting evidence.
* BUY / SELL are rejected when data quality is POOR or UNKNOWN.
* BUY / SELL are rejected when the model is UNTRAINED or DEGRADED.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.contracts.types import (
    IST,
    DataQualityStatus,
    Direction,
    ModelStatus,
    Timeframe,
    normalize_symbol,
)

CONTRACT_VERSION = "1.0.0"

_FLAG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware (naive timestamps are rejected)")
    return value.astimezone(IST)


class Evidence(BaseModel):
    """One piece of evidence behind an output."""

    model_config = ConfigDict(extra="forbid")

    factor: str = Field(min_length=1, description="Name of the factor, e.g. 'relative_volume'.")
    value: Optional[Union[float, str, bool]] = Field(default=None, description="Observed value.")
    detail: str = Field(default="", description="Plain-language explanation.")

    @field_validator("value")
    @classmethod
    def _finite(cls, v):
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("evidence value must be finite (no NaN / inf)")
        return v


class DataQuality(BaseModel):
    """Summary of the quality of the data an output was built on."""

    model_config = ConfigDict(extra="forbid")

    status: DataQualityStatus
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    n_observations: int = Field(default=0, ge=0)
    last_observation_time: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)

    @field_validator("last_observation_time")
    @classmethod
    def _aware(cls, v):
        return None if v is None else _require_aware(v, "last_observation_time")


class EngineOutput(BaseModel):
    """Validated result returned by every ULTRON engine."""

    model_config = ConfigDict(extra="forbid")

    engine_name: str = Field(min_length=1)
    engine_version: str = Field(pattern=r"^\d+\.\d+\.\d+([\-+][0-9A-Za-z.\-]+)?$")
    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    prediction_horizon: str = Field(pattern=r"^[1-9]\d*(m|h|d)$")
    direction: Direction
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    supporting_evidence: List[Evidence] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    data_quality: DataQuality
    model_status: ModelStatus

    contract_version: str = CONTRACT_VERSION
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_aware(cls, v):
        return _require_aware(v, "timestamp")

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v):
        return normalize_symbol(v)

    @field_validator("risk_flags")
    @classmethod
    def _flags(cls, v):
        for flag in v:
            if not _FLAG_RE.match(flag):
                raise ValueError(f"risk flag {flag!r} must be UPPER_SNAKE_CASE")
        return v

    @model_validator(mode="after")
    def _safety_rules(self):
        if self.direction in (Direction.BUY, Direction.SELL):
            if not self.supporting_evidence:
                raise ValueError("BUY/SELL requires at least one supporting_evidence item")
            if self.data_quality.status in (DataQualityStatus.POOR, DataQualityStatus.UNKNOWN):
                raise ValueError("BUY/SELL is not allowed when data quality is POOR or UNKNOWN")
            if self.model_status in (ModelStatus.UNTRAINED, ModelStatus.DEGRADED):
                raise ValueError("BUY/SELL is not allowed when model_status is UNTRAINED or DEGRADED")
        return self

    def to_json_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary (enums -> strings, datetimes -> ISO strings)."""
        return self.model_dump(mode="json")
