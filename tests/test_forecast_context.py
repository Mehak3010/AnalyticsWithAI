import pandas as pd

from anomaly_agent.forecast_context import add_forecast_context


class FakeForecaster:
    def __init__(self, config):
        pass

    def forecast(self, series_dict):
        idx = pd.DatetimeIndex(["2024-03-11", "2024-03-12"])
        return {
            ("sales", "total"): {
                "values": pd.Series([120.0, 121.0], index=idx),
                "lower": pd.Series([100.0, 101.0], index=idx),
                "upper": pd.Series([140.0, 141.0], index=idx),
                "model": "ets",
            }
        }


def test_attaches_first_future_forecast_only_to_matching_headline(monkeypatch):
    monkeypatch.setattr("anomaly_agent.forecast_context.Forecaster", FakeForecaster)
    anomalies = [
        {"metric": "sales", "segment": "total"},
        {"metric": "sales", "segment": "north"},
        {"metric": "orders", "segment": "total"},
    ]

    add_forecast_context(anomalies, {}, object())

    assert anomalies[0]["forecast_context"] == {
        "date": "2024-03-11",
        "value": 120.0,
        "lower": 100.0,
        "upper": 140.0,
        "model": "ets",
    }
    assert "forecast_context" not in anomalies[1]
    assert "forecast_context" not in anomalies[2]


def test_historical_anomaly_does_not_get_forecast_from_later_observations(monkeypatch):
    monkeypatch.setattr("anomaly_agent.forecast_context.Forecaster", FakeForecaster)
    series = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.DatetimeIndex(["2024-03-08", "2024-03-09", "2024-03-10"]),
    )
    anomalies = [{"metric": "sales", "segment": "total", "date": "2024-03-09"}]

    add_forecast_context(anomalies, {("sales", "total"): series}, object())

    assert "forecast_context" not in anomalies[0]


def test_latest_anomaly_gets_forecast_with_origin_and_horizon(monkeypatch):
    monkeypatch.setattr("anomaly_agent.forecast_context.Forecaster", FakeForecaster)
    series = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.DatetimeIndex(["2024-03-08", "2024-03-09", "2024-03-10"]),
    )
    anomalies = [{"metric": "sales", "segment": "total", "date": "2024-03-10"}]

    add_forecast_context(anomalies, {("sales", "total"): series}, object())

    assert anomalies[0]["forecast_context"]["origin_date"] == "2024-03-10"
    assert anomalies[0]["forecast_context"]["horizon_days"] == 1


def test_forecast_failure_does_not_block_alerts(monkeypatch):
    class BrokenForecaster:
        def __init__(self, config):
            pass

        def forecast(self, series_dict):
            raise RuntimeError("forecast failed")

    monkeypatch.setattr("anomaly_agent.forecast_context.Forecaster", BrokenForecaster)
    anomalies = [{"metric": "sales", "segment": "total"}]
    add_forecast_context(anomalies, {}, object())
    assert anomalies == [{"metric": "sales", "segment": "total"}]
