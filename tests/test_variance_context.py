from types import SimpleNamespace

import pandas as pd
import pytest

from anomaly_agent.variance_context import add_variance_context


def make_config():
    return SimpleNamespace(
        metrics=[SimpleNamespace(name="sales", type="additive")],
        dataset=SimpleNamespace(
            date_column="date",
            grain="daily",
            dimensions=["region"],
        ),
    )


def make_frame():
    rows = []
    # Four weekly baseline observations, plus the anomalous day.
    for date in ["2024-02-12", "2024-02-19", "2024-02-26", "2024-03-04"]:
        rows.append(
            {"date": pd.Timestamp(date), "region": "north", "quantity": 10, "sales": 100.0}
        )
    rows.append(
        {"date": pd.Timestamp("2024-03-11"), "region": "north", "quantity": 12, "sales": 144.0}
    )
    return pd.DataFrame(rows)


def test_attaches_reconciled_variance_bridge_to_total_sales_anomaly():
    anomalies = [{"date": "2024-03-11", "metric": "sales", "segment": "total"}]

    add_variance_context(anomalies, make_frame(), make_config())

    context = anomalies[0]["variance_context"]
    assert context["dimension"] == "region"
    assert context["baseline_revenue"] == pytest.approx(100.0)
    assert context["actual_revenue"] == pytest.approx(144.0)
    assert context["variance"] == pytest.approx(44.0)
    assert context["bridge"]["total"] == pytest.approx(context["variance"])
    assert (
        context["bridge"]["volume"]
        + context["bridge"]["mix"]
        + context["bridge"]["price"]
        == pytest.approx(context["variance"])
    )


def test_ignores_non_headline_and_non_sales_anomalies():
    anomalies = [
        {"date": "2024-03-11", "metric": "sales", "segment": "north"},
        {"date": "2024-03-11", "metric": "orders", "segment": "total"},
    ]

    add_variance_context(anomalies, make_frame(), make_config())

    assert all("variance_context" not in anomaly for anomaly in anomalies)


def test_missing_required_columns_fails_open():
    anomalies = [{"date": "2024-03-11", "metric": "sales", "segment": "total"}]

    add_variance_context(anomalies, pd.DataFrame({"date": [pd.Timestamp("2024-03-11")]}), make_config())

    assert "variance_context" not in anomalies[0]


def test_bad_anomaly_date_does_not_block_alert():
    anomalies = [{"date": "not-a-date", "metric": "sales", "segment": "total"}]

    add_variance_context(anomalies, make_frame(), make_config())

    assert anomalies == [{"date": "not-a-date", "metric": "sales", "segment": "total"}]
