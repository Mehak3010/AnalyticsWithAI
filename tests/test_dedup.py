from datetime import datetime, timedelta

import pytest

from anomaly_agent.store import AlertStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_alerts.db"
    return AlertStore(str(db_file))


def test_dedup_suppresses_within_cooldown(store):
    anomaly = {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": "high",
        "date": datetime.now(),
    }

    # 1. First alert
    to_alert1 = store.check_and_update_batch([anomaly], cooldown_days=3)
    assert len(to_alert1) == 1

    # Mark as sent
    fp = store.get_fingerprint(anomaly["metric"], anomaly["segment"], anomaly["direction"])
    store.mark_email_sent([fp])

    # 2. Repeat alert within cooldown
    to_alert2 = store.check_and_update_batch([anomaly], cooldown_days=3)
    assert len(to_alert2) == 0, "Repeat anomaly within cooldown should be suppressed"


def test_dedup_fires_after_cooldown(store):
    anomaly = {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": "high",
        "date": datetime.now(),
    }

    # 1. First alert
    store.check_and_update_batch([anomaly], cooldown_days=3)
    fp = store.get_fingerprint(anomaly["metric"], anomaly["segment"], anomaly["direction"])
    store.mark_email_sent([fp])

    # Manually backdate last_seen in DB
    past_date = datetime.now() - timedelta(days=5)
    with store._session() as conn:
        conn.execute(
            "UPDATE alerts SET last_seen = ? WHERE fingerprint = ?",
            (past_date.isoformat(), fp),
        )

    # 2. Repeat alert after cooldown
    to_alert2 = store.check_and_update_batch([anomaly], cooldown_days=3)
    assert len(to_alert2) == 1, "Repeat anomaly after cooldown should fire"


def test_dedup_fires_on_severity_change(store):
    anomaly_low = {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": "low",
        "date": datetime.now(),
    }

    anomaly_high = {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": "high",
        "date": datetime.now(),
    }

    # 1. Low severity alert
    store.check_and_update_batch([anomaly_low], cooldown_days=3)
    fp = store.get_fingerprint(anomaly_low["metric"], anomaly_low["segment"], anomaly_low["direction"])
    store.mark_email_sent([fp])

    # 2. High severity alert within cooldown
    to_alert2 = store.check_and_update_batch([anomaly_high], cooldown_days=3)
    assert len(to_alert2) == 1, "Alert should fire if severity changes, even within cooldown"


def test_unsent_alert_is_retried_not_lost(store):
    anomaly = {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": "high",
        "date": datetime.now(),
    }

    # 1. First alert, but delivery fails: never marked as sent
    assert len(store.check_and_update_batch([anomaly], cooldown_days=3)) == 1

    # 2. Next run: the alert must come back so the email can be retried
    assert len(store.check_and_update_batch([anomaly], cooldown_days=3)) == 1, "Unsent alert must be retried"
