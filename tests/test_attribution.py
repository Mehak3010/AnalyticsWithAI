import numpy as np
import pandas as pd
import pytest

from anomaly_agent.attribution import AttributionEngine, decompose_ratio
from anomaly_agent.config import Config
from anomaly_agent.narration.facts import to_llm_facts

TARGET = pd.Timestamp("2016-06-20")


def make_config() -> Config:
    return Config(
        dataset={
            "path": "x.csv",
            "date_column": "order_date",
            "grain": "daily",
            "segment_grain": "weekly",
            "dimensions": ["market", "channel"],
        },
        metrics=[
            {
                "name": "sales",
                "column": "sales",
                "type": "additive",
                "aggregation": "sum",
                "priority": "high",
                "good_direction": "up",
            },
            {
                "name": "profit_margin",
                "type": "ratio",
                "numerator": {"column": "profit", "aggregation": "sum"},
                "denominator": {"column": "sales", "aggregation": "sum"},
                "priority": "high",
                "good_direction": "up",
            },
            {
                "name": "late_rate",
                "type": "ratio",
                "numerator": {"column": "late_flag", "aggregation": "sum"},
                "denominator": {"aggregation": "count"},
                "priority": "high",
                "good_direction": "down",
            },
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
            "email": {"to": ["a@b.c"], "subject_prefix": "[T]"},
        },
        llm={"provider": "groq", "model": "m", "max_retries": 0},
    )


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Three markets x two channels, 10 order lines each per day, ~2% noise."""
    rng = np.random.default_rng(7)
    rows = []
    for day in pd.date_range("2016-04-01", "2016-06-30"):
        for market in ("A", "B", "C"):
            for channel in ("X", "Y"):
                for _ in range(10):
                    sales = 100 * rng.normal(1.0, 0.02)
                    rows.append(
                        {
                            "order_date": day,
                            "market": market,
                            "channel": channel,
                            "sales": sales,
                            "profit": sales * 0.11,
                            "late_flag": int(rng.random() < 0.1),
                        }
                    )
    return pd.DataFrame(rows)


def anomaly(metric):
    return {"metric": metric, "segment": "total", "date": TARGET, "direction": "up"}


def rows_of(df, market):
    return (df.order_date == TARGET) & (df.market == market)


# ---------------------------------------------------------------- ratio math


def S(*values):
    return pd.Series(values, index=["A", "B"], dtype=float)


def test_pure_mix_shift_is_all_mix_effect():
    # A: 10/100 = 10%, B: 30/100 = 30%. Volume moves from A to B, ratios unchanged.
    result = decompose_ratio(num1=S(5, 45), den1=S(50, 150), num0=S(10, 30), den0=S(100, 100))
    assert result["R0"] == pytest.approx(0.20)
    assert result["R1"] == pytest.approx(0.25)
    t = result["table"]
    assert t["mix"].sum() == pytest.approx(0.05)
    assert t["rate"].sum() == pytest.approx(0.0)
    assert t["mix"]["A"] == pytest.approx(0.025) and t["mix"]["B"] == pytest.approx(0.025)


def test_pure_rate_change_is_all_rate_effect():
    # Volumes unchanged; A's ratio rises from 10% to 20%.
    result = decompose_ratio(num1=S(20, 30), den1=S(100, 100), num0=S(10, 30), den0=S(100, 100))
    t = result["table"]
    assert t["mix"].sum() == pytest.approx(0.0)
    assert t["rate"].sum() == pytest.approx(0.05)
    assert t["rate"]["A"] == pytest.approx(0.05) and t["rate"]["B"] == pytest.approx(0.0)


def test_mix_plus_rate_equals_total_change_on_random_data():
    rng = np.random.default_rng(1)
    idx = list("ABCDE")
    for _ in range(50):
        den0 = pd.Series(rng.uniform(10, 100, 5), index=idx)
        den1 = pd.Series(rng.uniform(10, 100, 5), index=idx)
        num0 = den0 * pd.Series(rng.uniform(0.05, 0.5, 5), index=idx)
        num1 = den1 * pd.Series(rng.uniform(0.05, 0.5, 5), index=idx)
        r = decompose_ratio(num1, den1, num0, den0)
        assert r["table"]["effect"].sum() == pytest.approx(r["total_change"])


def test_identity_holds_with_a_segment_that_appears_or_disappears():
    # B had no volume in the baseline; A has none now.
    r = decompose_ratio(num1=S(0, 30), den1=S(0, 100), num0=S(10, 0), den0=S(100, 0))
    assert r["table"]["effect"].sum() == pytest.approx(r["total_change"])


def test_zero_total_denominator_returns_none():
    assert decompose_ratio(S(0, 0), S(0, 0), S(1, 1), S(10, 10)) is None


# ------------------------------------------------------------ additive metric


def test_additive_attribution_finds_the_injected_segment(df):
    d = df.copy()
    d.loc[rows_of(d, "C"), "sales"] *= 4
    result = AttributionEngine(make_config()).attribute(anomaly("sales"), d)

    assert result["primary_dimension"] == "market"
    top = result["contributors"][0]
    assert top["segment"] == "market=C"
    assert top["share"] > 0.9
    assert result["coverage"] >= 0.8
    assert "market=C" in result["drivers"][0] and "of the move" in result["drivers"][0]
    # channel is evenly affected, so it must not be chosen as the story
    assert result["by_dimension"]["channel"]["contributors"][0]["share"] < 0.6


def test_shares_never_exceed_the_number_of_contributors_cap(df):
    d = df.copy()
    d.loc[d.order_date == TARGET, "sales"] *= 2  # every segment moves equally
    result = AttributionEngine(make_config()).attribute(anomaly("sales"), d)
    assert len(result["contributors"]) <= 3


def test_share_is_never_above_100_percent_even_when_others_offset(df):
    d = df.copy()
    d.loc[rows_of(d, "C"), "sales"] *= 4
    d.loc[rows_of(d, "A"), "sales"] *= 0.9  # a small move the other way
    result = AttributionEngine(make_config()).attribute(anomaly("sales"), d)
    assert all(0 < c["share"] <= 1 for c in result["contributors"])
    assert result["contributors"][0]["net_share"] > result["contributors"][0]["share"]


def test_non_total_anomaly_is_not_attributed(df):
    a = {**anomaly("sales"), "segment": "market=A"}
    assert AttributionEngine(make_config()).attribute(a, df) == {}


def test_not_enough_history_returns_empty(df):
    short = df[df.order_date >= "2016-06-18"]
    assert AttributionEngine(make_config()).attribute(anomaly("sales"), short) == {}


# --------------------------------------------------------------- ratio metric


def test_margin_drop_in_one_market_is_a_rate_effect_there(df):
    d = df.copy()
    d.loc[rows_of(d, "A"), "profit"] = 0.0
    result = AttributionEngine(make_config()).attribute(anomaly("profit_margin"), d)

    assert result["metric_type"] == "ratio"
    assert result["primary_dimension"] == "market"
    top = result["contributors"][0]
    assert top["segment"] == "market=A"
    assert abs(top["rate"]) > abs(top["mix"])
    assert "Mostly a rate effect" in result["note"]
    assert "mainly a rate change" in result["drivers"][0]
    best = result["by_dimension"]["market"]
    assert best["mix_effect"] + best["rate_effect"] == pytest.approx(best["total_change"], abs=1e-5)


def test_late_rate_with_count_denominator(df):
    d = df.copy()
    d.loc[rows_of(d, "B"), "late_flag"] = 1
    result = AttributionEngine(make_config()).attribute(anomaly("late_rate"), d)
    assert result["contributors"][0]["segment"] == "market=B"
    assert "rate effect" in result["note"]


def test_numerator_and_denominator_changes_are_reported(df):
    d = df.copy()
    d.loc[rows_of(d, "C"), "late_flag"] = 1
    result = AttributionEngine(make_config()).attribute(anomaly("late_rate"), d)
    m = result["by_dimension"]["market"]
    assert m["numerator_change"] > 1.0  # late orders more than doubled
    assert abs(m["denominator_change"]) < 0.01  # order count unchanged
    assert "The numerator changed by" in result["note"]


# ------------------------------------------------------------------ narration


def test_driver_note_reaches_the_llm_bundle():
    facts = {
        "metric": "late_rate",
        "segment": "total",
        "date": "2016-06-20",
        "direction": "up",
        "observed": 0.2,
        "expected": 0.1,
        "rel_deviation": 1.0,
        "is_good": False,
        "drivers": ["market=B (80% of the change, mainly a rate change)"],
        "driver_note": "Mostly a rate effect (95%): the ratio changed within segments.",
    }
    slim = to_llm_facts(facts)
    assert slim["driver_note"].startswith("Mostly a rate effect")
    assert "driver_note" not in to_llm_facts({**facts, "driver_note": ""})
