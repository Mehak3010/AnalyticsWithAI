"""Robust time-series forecasting with an ETS model and seasonal-naive fallback."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import Config

logger = logging.getLogger(__name__)


class Forecaster:
    """Forecast headline KPI series and return prediction intervals."""

    def __init__(self, config: Config):
        self.config = config

    def forecast(
        self, series_dict: dict[tuple[str, str], pd.Series]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        forecasts: dict[tuple[str, str], dict[str, Any]] = {}
        for (metric, segment), raw_series in series_dict.items():
            # Segment-level forecasts are deliberately omitted; forecast headline KPIs only.
            if segment != "total":
                continue

            series = self._prepare_series(raw_series)
            if len(series) < self.config.detection.min_history:
                logger.info(
                    "Skipping forecast for %s: %d observations, need %d",
                    metric,
                    len(series),
                    self.config.detection.min_history,
                )
                continue

            try:
                forecasts[(metric, segment)] = self._ets_forecast(series)
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                logger.warning("ETS forecast failed for %s; using seasonal-naive fallback: %s", metric, exc)
                forecasts[(metric, segment)] = self._seasonal_naive(series)

        return forecasts

    @staticmethod
    def _prepare_series(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Each forecast input must be a pandas Series.")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("Forecast series must use a DatetimeIndex.")
        if series.empty:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        if series.index.has_duplicates:
            raise ValueError("Forecast series index must not contain duplicate dates.")

        prepared = pd.to_numeric(series, errors="coerce").sort_index().astype(float)
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        if prepared.notna().sum() == 0:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        return prepared.ffill().bfill()

    def _ets_forecast(self, series: pd.Series) -> dict[str, Any]:
        seasonal_period = self.config.detection.seasonal_period
        if seasonal_period < 2 or len(series) < 2 * seasonal_period:
            raise ValueError("ETS requires at least two complete seasonal cycles.")

        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)

        horizon = self.config.forecast.horizon_days
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least one period.")
        values = model.forecast(horizon)
        residuals = pd.Series(model.resid).replace([np.inf, -np.inf], np.nan).dropna()
        scale = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
        if not np.isfinite(scale):
            scale = 0.0
        z = self._z_value()
        lower = values - z * scale
        upper = values + z * scale
        return {
            "values": values,
            "lower": lower,
            "upper": upper,
            "model": "ets",
        }

    def _seasonal_naive(self, series: pd.Series) -> dict[str, Any]:
        horizon = self.config.forecast.horizon_days
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least one period.")
        if series.empty:
            raise ValueError("Cannot forecast an empty series.")

        seasonal_period = max(1, self.config.detection.seasonal_period)
        cycle = series.iloc[-min(seasonal_period, len(series)):].to_numpy(dtype=float)
        repeated = np.resize(cycle, horizon)
        dates = self._future_index(series.index, horizon)
        values = pd.Series(repeated, index=dates, name=series.name)

        differences = series - series.shift(min(seasonal_period, len(series) - 1 or 1))
        differences = differences.replace([np.inf, -np.inf], np.nan).dropna()
        scale = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
        if not np.isfinite(scale):
            scale = 0.0
        margin = self._z_value() * scale
        return {
            "values": values,
            "lower": values - margin,
            "upper": values + margin,
            "model": "seasonal_naive",
        }

    def _z_value(self) -> float:
        interval = float(self.config.forecast.interval)
        if not 0.0 < interval < 1.0:
            raise ValueError("Forecast interval must be between 0 and 1, exclusive.")
        return float(norm.ppf((1.0 + interval) / 2.0))

    @staticmethod
    def _future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
        frequency = index.freq or pd.infer_freq(index)
        if frequency is not None:
            offset = pd.tseries.frequencies.to_offset(frequency)
        else:
            # The project monitors daily KPIs; irregular dates default to daily forecasts.
            offset = pd.tseries.frequencies.to_offset("D")
        start = index[-1] + offset
        return pd.date_range(start=start, periods=horizon, freq=offset)
