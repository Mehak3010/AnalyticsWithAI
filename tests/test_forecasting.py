import numpy as np
import pandas as pd
import pytest

from anomaly_agent.config import Config
from anomaly_agent.forecasting import Forecaster


def make_config() -> Config:
    return Config(
        dataset={
            "path": "unused.csv",
            "date_column": "date",
            "grain": "daily",
            "segment_grain": "weekly",
            "dimensions": ["region"],
        },
        metrics=[
            {
                "name": "sales",
                "column": "sales",
                "type": "additive",
                "aggregation": "sum",
                "priority": "high",
                "good_direction": "up",
            }
        ],
        forecast={
            "method": "ets",
            "horizon_days": 14,
            "interval": 0.95,
            "backtest": {"folds": 2, "step_days": 14},
        },
        detection={
            "seasonal_period": 7,
            "min_history": 28,
            "z_threshold": 3.5,
            "persistence": {"required": 2, "window": 3},
            "min_relative_change": 0.05,
        },
        alerting={
            "min_severity": "medium",
            "cooldown_days": 3,
            "email": {"to": ["test@example.com"], "subject_prefix": "[Test]"},
        },
        llm={"provider": "groq", "model": "test", "max_retries": 0},
    )


def make_series(n: int = 70) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    values = 100 + np.arange(n) * 0.2 + 5 * np.sin(2 * np.pi * index.dayofweek / 7)
    return pd.Series(values, index=index, name="sales")


def test_forecast_returns_horizon_and_ordered_intervals():
    forecaster = Forecaster(make_config())
    result = forecaster.forecast({("sales", "total"): make_series()})[("sales", "total")]

    assert len(result["values"]) == 14
    assert result["values"].index[0] == pd.Timestamp("2024-03-11")
    assert (result["lower"] <= result["values"]).all()
    assert (result["values"] <= result["upper"]).all()
    assert result["model"] in {"ets", "seasonal_naive"}


def test_forecast_skips_segment_series_and_insufficient_history():
    forecaster = Forecaster(make_config())
    result = forecaster.forecast(
        {
            ("sales", "north"): make_series(),
            ("sales", "total"): make_series(10),
        }
    )
    assert result == {}


def test_seasonal_naive_repeats_last_cycle_and_has_future_dates():
    forecaster = Forecaster(make_config())
    series = make_series(28)
    result = forecaster._seasonal_naive(series)

    assert len(result["values"]) == 14
    assert result["values"].iloc[:7].to_list() == pytest.approx(series.iloc[-7:].to_list())
    assert result["values"].index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert (result["lower"] <= result["values"]).all()
    assert (result["values"] <= result["upper"]).all()
    assert result["model"] == "seasonal_naive"


def test_forecast_cleans_missing_values_before_fitting():
    forecaster = Forecaster(make_config())
    series = make_series()
    series.iloc[10] = np.nan
    result = forecaster.forecast({("sales", "total"): series})
    assert ("sales", "total") in result


def test_forecast_rejects_duplicate_dates():
    forecaster = Forecaster(make_config())
    series = make_series()
    series.index = list(series.index[:-1]) + [series.index[-2]]
    with pytest.raises(ValueError, match="duplicate"):
        forecaster.forecast({("sales", "total"): series})


def test_forecast_rejects_invalid_interval():
    config = make_config()
    config.forecast.interval = 1.0
    forecaster = Forecaster(config)
    with pytest.raises(ValueError, match="between 0 and 1"):
        forecaster._z_value()
