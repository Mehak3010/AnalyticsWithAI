from typing import Any

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .config import Config


class Forecaster:
    def __init__(self, config: Config):
        self.config = config

    def forecast(self, series_dict: dict[tuple[str, str], pd.Series]) -> dict[tuple[str, str], dict[str, Any]]:
        forecasts = {}
        for (metric, segment), series in series_dict.items():
            # Only forecast headline series (total) as per specification
            if segment != "total":
                continue

            if len(series.dropna()) < self.config.detection.min_history:
                continue

            series = series.ffill().bfill()

            # Fit ETS
            try:
                # Determine seasonality
                seasonal_period = self.config.detection.seasonal_period

                # Holt-Winters ETS
                # Weekly grain: use damped trend with no seasonal term unless 2+ years exist
                # (DataCo is daily, so we use seasonal=add/mul)
                model = ExponentialSmoothing(
                    series,
                    seasonal_periods=seasonal_period,
                    trend="add",
                    seasonal="add",
                    damped_trend=True,
                ).fit()

                horizon = self.config.forecast.horizon_days
                forecast_values = model.forecast(horizon)

                # Simple prediction intervals (95% = 1.96 * std_dev of residuals)
                residuals = model.resid
                std_dev = residuals.std()
                z = 1.96  # 95% interval

                lower = forecast_values - z * std_dev
                upper = forecast_values + z * std_dev

                forecasts[(metric, segment)] = {
                    "values": forecast_values,
                    "lower": lower,
                    "upper": upper,
                    "model": "ets",
                }
            except Exception as e:
                print(f"Forecasting error for {metric}: {e}")
                # Fallback to seasonal naive
                forecasts[(metric, segment)] = self._seasonal_naive(series)

        return forecasts

    def _seasonal_naive(self, series: pd.Series) -> dict[str, Any]:
        seasonal_period = self.config.detection.seasonal_period
        horizon = self.config.forecast.horizon_days

        # Take the last 'seasonal_period' values and repeat them
        last_cycle = series.iloc[-seasonal_period:]
        forecast_values = pd.concat([last_cycle] * (horizon // seasonal_period + 1)).iloc[:horizon]

        # Dates for forecast
        last_date = series.index[-1]
        freq = series.index.freq if series.index.freq else "D"
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq=freq)
        forecast_values.index = forecast_dates

        # Simple interval for naive
        std_dev = series.diff(seasonal_period).std()
        z = 1.96

        return {
            "values": forecast_values,
            "lower": forecast_values - z * std_dev,
            "upper": forecast_values + z * std_dev,
            "model": "seasonal_naive",
        }
