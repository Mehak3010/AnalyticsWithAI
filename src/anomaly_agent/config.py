from typing import Literal

import yaml
from pydantic import BaseModel


class MetricAggregation(BaseModel):
    column: str | None = None
    aggregation: str


class MetricConfig(BaseModel):
    name: str
    column: str | None = None
    type: Literal["additive", "ratio"]
    aggregation: str | None = None
    numerator: MetricAggregation | None = None
    denominator: MetricAggregation | None = None
    priority: Literal["high", "medium", "low"]
    good_direction: Literal["up", "down", "neutral"]


class ForecastConfig(BaseModel):
    method: Literal["ets", "seasonal_naive"]
    horizon_days: int
    interval: float
    backtest: dict[str, int]


class DetectionConfig(BaseModel):
    seasonal_period: int
    min_history: int
    z_threshold: float
    persistence: dict[str, int]
    min_relative_change: float
    # Floor on the robust scale, as a fraction of the series level. Stops a near-zero
    # MAD (sparse or flat series) from producing enormous z-scores.
    mad_floor_pct: float = 0.02
    # A single point this extreme alerts without persistence. Default: 3 x z_threshold.
    spike_override_z: float | None = None


class AlertingConfig(BaseModel):
    min_severity: Literal["low", "medium", "high", "critical"]
    cooldown_days: int
    email: dict[str, str | list[str]]
    # Only anomalies in the last N periods of a series can raise an alert.
    alert_window: int = 3
    # Only the headline ("total") series raises alerts; segments explain them.
    headline_only: bool = True


class LLMConfig(BaseModel):
    provider: str
    model: str
    max_retries: int


class DatasetConfig(BaseModel):
    path: str
    date_column: str
    grain: str
    segment_grain: str
    dimensions: list[str]
    # Optional monitoring window (YYYY-MM-DD); rows outside it are ignored at load time.
    start_date: str | None = None
    end_date: str | None = None


class Config(BaseModel):
    dataset: DatasetConfig
    metrics: list[MetricConfig]
    forecast: ForecastConfig
    detection: DetectionConfig
    alerting: AlertingConfig
    llm: LLMConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
