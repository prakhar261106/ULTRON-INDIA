"""Central configuration loader for ULTRON.

``load_config()`` reads ``config/ultron_config.yaml`` and validates every value, so a typo in
the YAML fails loudly instead of silently changing behaviour.

Secrets: use :func:`get_secret`. It reads environment variables (or Colab Secrets when running
in Google Colab). Secrets are never stored in the config file and never logged.
"""
from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.contracts.types import Timeframe

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "ultron_config.yaml"

VALID_WEIGHT_KEYS = {
    "avg_daily_traded_value",
    "liquidity_bar_value",
    "relative_volume",
    "atr_pct",
    "volatility_pct",
    "gap_pct",
    "momentum_pct",
    "relative_strength_pct",
    "sector_strength_pct",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketConfig(_Strict):
    exchange: str = "NSE"
    timezone: str = "Asia/Kolkata"
    session_open: time = time(9, 15)
    session_close: time = time(15, 30)
    holidays: List[date] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if self.timezone != "Asia/Kolkata":
            raise ValueError("ULTRON uses Asia/Kolkata for all timestamps")
        if self.session_close <= self.session_open:
            raise ValueError("session_close must be after session_open")
        return self

    @property
    def session_minutes(self) -> int:
        return (self.session_close.hour * 60 + self.session_close.minute) - (
            self.session_open.hour * 60 + self.session_open.minute
        )


class DataProviderConfig(_Strict):
    name: Optional[str] = None
    credentials_env_vars: List[str] = Field(default_factory=list)
    require_adjusted_prices: bool = True


class DataQualityConfig(_Strict):
    max_missing_candle_ratio: float = Field(default=0.10, ge=0, le=1)
    max_zero_volume_ratio: float = Field(default=0.20, ge=0, le=1)
    max_staleness_bars: int = Field(default=2, ge=0)
    max_unexplained_missing_days: int = Field(default=2, ge=0)


class StockSelectionConfig(_Strict):
    top_n: int = Field(default=10, ge=1)
    prediction_horizon: str = Field(default="1d", pattern=r"^[1-9]\d*(m|h|d)$")
    lookback_sessions: int = Field(default=20, ge=2)
    min_sessions: int = Field(default=10, ge=2)
    atr_period: int = Field(default=14, ge=2)
    momentum_lookback_bars: int = Field(default=20, ge=1)
    min_price_inr: float = Field(default=50.0, ge=0)
    min_avg_daily_volume: float = Field(default=100_000, ge=0)
    min_avg_daily_traded_value_inr: float = Field(default=5e8, ge=0)
    min_feature_coverage: float = Field(default=0.8, ge=0, le=1)
    gap_flag_pct: float = Field(default=5.0, gt=0)
    low_relative_volume_flag: float = Field(default=0.5, ge=0)
    high_volatility_percentile_flag: float = Field(default=0.9, ge=0, le=1)
    weights: Dict[str, float]

    @field_validator("weights")
    @classmethod
    def _weights(cls, w):
        unknown = set(w) - VALID_WEIGHT_KEYS
        if unknown:
            raise ValueError(f"unknown ranking weight(s): {sorted(unknown)}")
        if any(v < 0 for v in w.values()):
            raise ValueError("ranking weights must be >= 0")
        if sum(w.values()) <= 0:
            raise ValueError("at least one ranking weight must be > 0")
        return w

    @model_validator(mode="after")
    def _sessions(self):
        if self.min_sessions > self.lookback_sessions:
            raise ValueError("min_sessions cannot exceed lookback_sessions")
        return self

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items() if v > 0}


class PathsConfig(_Strict):
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    features: str = "data/features"
    logs: str = "logs"


class UltronConfig(_Strict):
    market: MarketConfig = Field(default_factory=MarketConfig)
    timeframes: List[Timeframe] = Field(default_factory=lambda: list(Timeframe))
    data_provider: DataProviderConfig = Field(default_factory=DataProviderConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    stock_selection: StockSelectionConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def path(self, name: str) -> Path:
        """Absolute path of a configured folder, e.g. ``cfg.path('processed_data')``."""
        return REPO_ROOT / getattr(self.paths, name)


def load_config(path: str | Path | None = None) -> UltronConfig:
    """Load and validate the YAML configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return UltronConfig(**raw)


def get_secret(name: str, required: bool = False) -> Optional[str]:
    """Read a secret from an environment variable, or from Colab Secrets if available.

    The value is returned to the caller only; it is never printed or logged here.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:  # Google Colab Secrets (only exists inside Colab)
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    if required:
        raise RuntimeError(
            f"Secret {name!r} is not set. Set it as an environment variable or in Colab Secrets."
        )
    return None
