from typing import List, Optional, Union, Dict, Literal
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path

class MetricAggregation(BaseModel):
    column: Optional[str] = None
    aggregation: str

class MetricConfig(BaseModel):
    name: str
    column: Optional[str] = None
    type: Literal["additive", "ratio"]
    aggregation: Optional[str] = None
    numerator: Optional[MetricAggregation] = None
    denominator: Optional[MetricAggregation] = None
    priority: Literal["high", "medium", "low"]
    good_direction: Literal["up", "down", "neutral"]

class ForecastConfig(BaseModel):
    method: Literal["ets", "seasonal_naive"]
    horizon_days: int
    interval: float
    backtest: Dict[str, int]

class DetectionConfig(BaseModel):
    seasonal_period: int
    min_history: int
    z_threshold: float
    persistence: Dict[str, int]
    min_relative_change: float

class AlertingConfig(BaseModel):
    min_severity: Literal["low", "medium", "high", "critical"]
    cooldown_days: int
    email: Dict[str, Union[str, List[str]]]

class LLMConfig(BaseModel):
    provider: str
    model: str
    max_retries: int

class DatasetConfig(BaseModel):
    path: str
    date_column: str
    grain: str
    segment_grain: str
    dimensions: List[str]

class Config(BaseModel):
    dataset: DatasetConfig
    metrics: List[MetricConfig]
    forecast: ForecastConfig
    detection: DetectionConfig
    alerting: AlertingConfig
    llm: LLMConfig

def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
