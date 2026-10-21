"""Attach clearly dated forward-looking forecast context to headline anomalies."""

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
    """Attach a forward forecast only when it is temporally relevant.

    Forecast context is not the historical baseline used by the detector. If an
    anomaly has a date, only attach a forecast when that date is the final observed
    date in the corresponding headline series. This prevents a forecast fitted on
    later data from being presented as if it were available at an earlier anomaly.
    Forecast failures are non-fatal to alerting.
    """
    if not any(anomaly.get("segment") == "total" for anomaly in anomalies):
        return
    try:
        forecasts = Forecaster(config).forecast(series_dict)
    except Exception as exc:
        logger.warning("Forecast context unavailable: %s: %s", type(exc).__name__, exc)
        return

    for anomaly in anomalies:
        if anomaly.get("segment") != "total":
            continue
        metric = anomaly.get("metric")
        forecast = forecasts.get((metric, "total"))
        if not forecast:
            continue
        try:
            values = forecast["values"]
            lower = forecast["lower"]
            upper = forecast["upper"]
            if values.empty:
                continue
            if not values.index.equals(lower.index) or not values.index.equals(upper.index):
                continue

            series = series_dict.get((metric, "total"))
            origin = None
            if series is not None and not series.empty:
                observed = pd.to_numeric(series, errors="coerce")
                observed = observed.replace([float("inf"), float("-inf")], float("nan")).dropna()
                if observed.empty:
                    continue
                origin = pd.Timestamp(observed.index.max()).normalize()

            forecast_date = pd.Timestamp(values.index[0]).normalize()
            if origin is not None:
                if forecast_date <= origin:
                    continue
                anomaly_date_raw = anomaly.get("date")
                if anomaly_date_raw is not None:
                    anomaly_date = pd.Timestamp(anomaly_date_raw).normalize()
                    # Historical anomalies cannot borrow forecasts fitted on later data.
                    if anomaly_date != origin:
                        continue

            value = float(values.iloc[0])
            low = float(lower.iloc[0])
            high = float(upper.iloc[0])
            if not all(pd.notna(number) for number in (value, low, high)):
                continue
            if low > value or value > high:
                continue

            context = {
                "date": forecast_date.date().isoformat(),
                "value": value,
                "lower": low,
                "upper": high,
                "model": str(forecast.get("model", "forecast")),
            }
            if origin is not None:
                context["origin_date"] = origin.date().isoformat()
                context["horizon_days"] = int((forecast_date - origin).days)
            anomaly["forecast_context"] = context
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Invalid forecast context for %s: %s", metric, exc)
