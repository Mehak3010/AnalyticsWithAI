"""Attach forward-looking forecast context to detected anomalies."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import Config
from .forecasting import Forecaster

logger = logging.getLogger(__name__)


def add_forecast_context(
    anomalies: list[dict[str, Any]],
    series_dict: dict[tuple[str, str], pd.Series],
    config: Config,
) -> None:
    """Attach the first future forecast point to matching headline anomalies.

    This is forward-looking context, not a replacement for the historical expected
    value used by the detector. Forecast failures are non-fatal to alerting.
    """
    if not anomalies:
        return
    try:
        forecasts = Forecaster(config).forecast(series_dict)
    except Exception as exc:
        logger.warning("Forecast context unavailable: %s: %s", type(exc).__name__, exc)
        return

    for anomaly in anomalies:
        if anomaly.get("segment") != "total":
            continue
        forecast = forecasts.get((anomaly.get("metric"), "total"))
        if not forecast:
            continue
        try:
            values, lower, upper = forecast["values"], forecast["lower"], forecast["upper"]
            if values.empty:
                continue
            date = pd.Timestamp(values.index[0]).date().isoformat()
            value, low, high = float(values.iloc[0]), float(lower.iloc[0]), float(upper.iloc[0])
            if not all(pd.notna(x) for x in (value, low, high)) or low > value or value > high:
                continue
            anomaly["forecast_context"] = {
                "date": date, "value": value, "lower": low, "upper": high,
                "model": str(forecast.get("model", "forecast")),
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Invalid forecast context for %s: %s", anomaly.get("metric"), exc)
