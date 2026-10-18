import numpy as np
import pandas as pd
import pytest

from anomaly_agent.config import Config
from anomaly_agent.detection import Detector
from anomaly_agent.metrics import MetricEngine
from anomaly_agent.severity import SeverityScorer
from anomaly_agent.store import AlertStore


def make_config() -> Config:
    return Config(
        dataset={"path": "x.csv", "date_column": "order_date", "grain": "daily",
                 "segment_grain": "weekly", "dimensions": ["market"]},
        metrics=[
            {"name": "sales", "column": "sales", "type": "additive", "aggregation": "sum",
             "priority": "high", "good_direction": "up"},
            {"name": "profit_margin", "type": "ratio",
             "numerator": {"column": "profit", "aggregation": "sum"},
             "denominator": {"column": "sales", "aggregation": "sum"},
             "priority": "high", "good_direction": "up"},
        ],
        forecast={"method": "ets", "horizon_days": 14, "interval": 0.95, "backtest": {"folds": 2, "step_days": 14}},
        detection={"seasonal_period": 7, "min_history": 28, "z_threshold": 3.5,
                   "persistence": {"required": 2, "window": 3}, "min_relative_change": 0.05},
        alerting={"min_severity": "medium", "cooldown_days": 3, "email": {"to": ["a@b.c"], "subject_prefix": "[T]"}},
        llm={"provider": "groq", "model": "m", "max_retries": 0},
    )


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Two years of synthetic daily data with a weekly pattern and ~4% daily noise."""
    rng = np.random.default_rng(42)
    rows = []
    for day in pd.date_range("2015-01-01", "2016-12-31"):
        weekly = 1.0 + 0.2 * np.sin(2 * np.pi * day.dayofweek / 7)
        for market in ("A", "B"):
            sales = 10_000 * weekly * rng.normal(1.0, 0.04)
            rows.append({"order_date": day, "market": market, "sales": sales,
                         "profit": sales * rng.normal(0.11, 0.004)})
    return pd.DataFrame(rows)


def headline(df, cfg):
    return {k: v for k, v in MetricEngine(cfg).process(df).items() if k[1] == "total"}


def alerts_for(df, cfg, as_of, metric="sales"):
    found = Detector(cfg).detect(headline(df, cfg), as_of=pd.Timestamp(as_of), window=3)
    return [a for a in found if a["metric"] == metric]


def scaled(df, days, factor):
    out = df.copy()
    out.loc[out.order_date.isin(pd.to_datetime(days)), "sales"] *= factor
    return out


def test_ratio_is_recomputed_from_components_not_averaged(df):
    cfg = make_config()
    series = MetricEngine(cfg).process(df)[("profit_margin", "total")]
    day = pd.Timestamp("2016-03-01")
    d = df[df.order_date == day]
    assert series.loc[day] == pytest.approx(d.profit.sum() / d.sales.sum())


def test_clean_data_is_quiet(df):
    assert alerts_for(df, make_config(), "2016-06-17") == []


def test_spike_is_detected_on_its_date(df):
    found = alerts_for(scaled(df, ["2016-06-15"], 3), make_config(), "2016-06-17")
    assert [a["date"] for a in found] == [pd.Timestamp("2016-06-15")]
    assert found[0]["direction"] == "up"


def test_one_day_blip_is_suppressed(df):
    assert alerts_for(scaled(df, ["2016-03-10"], 1.2), make_config(), "2016-03-12") == []


def test_three_day_shift_is_detected_without_a_mirror_alert(df):
    days = ["2016-09-05", "2016-09-06", "2016-09-07"]
    found = alerts_for(scaled(df, days, 1.2), make_config(), "2016-09-08")
    assert found, "a sustained 3-day shift must alert"
    assert all(a["direction"] == "up" for a in found)  # no opposite-direction echo
    assert all(pd.Timestamp("2016-09-05") <= a["date"] <= pd.Timestamp("2016-09-07") for a in found)


def test_severity_increases_with_deviation():
    scorer = SeverityScorer(make_config())
    mild = {"z_score": 3.6, "rel_deviation": 0.06, "priority": "low"}
    big = {"z_score": 12.0, "rel_deviation": 0.6, "priority": "high"}
    order = ["low", "medium", "high", "critical"]
    assert order.index(scorer.score(big)) > order.index(scorer.score(mild))


def _anomaly(date, severity="medium"):
    return {"metric": "sales", "segment": "total", "direction": "up",
            "severity": severity, "date": pd.Timestamp(date)}


def test_dedup_suppresses_repeat_but_not_escalation(tmp_path):
    store = AlertStore(str(tmp_path / "alerts.db"))
    assert len(store.check_and_update_batch([_anomaly("2016-06-01")], 3)) == 1
    assert store.check_and_update_batch([_anomaly("2016-06-01")], 3) == []   # rerun
    assert store.check_and_update_batch([_anomaly("2016-06-02")], 3) == []   # inside cooldown
    assert len(store.check_and_update_batch([_anomaly("2016-06-02", "high")], 3)) == 1  # escalated
    assert len(store.check_and_update_batch([_anomaly("2016-06-20")], 3)) == 1  # after cooldown


def test_cli_imports():
    import anomaly_agent.cli  # noqa: F401
