import sqlite3
from datetime import datetime

from anomaly_agent.store import AlertStore

FP = AlertStore(":memory:").get_fingerprint("sales", "total", "up")


def anomaly(day=18, severity="medium"):
    return {
        "metric": "sales",
        "segment": "total",
        "direction": "up",
        "severity": severity,
        "date": datetime(2016, 9, day),
    }


def rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT fingerprint, last_seen, severity FROM alerts").fetchall()
    finally:
        conn.close()


def test_memory_store_persists_across_calls():
    store = AlertStore(":memory:")
    assert len(store.check_and_update_batch([anomaly()], 3)) == 1
    store.mark_email_sent([FP])
    assert store.check_and_update_batch([anomaly()], 3) == []


def test_real_run_is_idempotent(tmp_path):
    db = str(tmp_path / "alerts.db")
    first = AlertStore(db)
    assert len(first.check_and_update_batch([anomaly()], 3)) == 1
    first.mark_email_sent([FP])
    assert AlertStore(db).check_and_update_batch([anomaly()], 3) == []


def test_failed_send_is_retried_next_run(tmp_path):
    db = str(tmp_path / "alerts.db")
    assert len(AlertStore(db).check_and_update_batch([anomaly()], 3)) == 1  # email "failed": never marked sent
    assert len(AlertStore(db).check_and_update_batch([anomaly()], 3)) == 1  # retried, not lost
    store = AlertStore(db)
    store.mark_email_sent([FP])
    assert AlertStore(db).check_and_update_batch([anomaly()], 3) == []  # delivered, now suppressed


def test_retry_does_not_move_last_seen_back(tmp_path):
    db = str(tmp_path / "alerts.db")
    AlertStore(db).check_and_update_batch([anomaly(25)], 3)
    AlertStore(db).check_and_update_batch([anomaly(18)], 3)  # older replay of an unsent alert
    assert rows(db)[0][1].startswith("2016-09-25")


def test_dry_run_creates_no_file(tmp_path):
    db = tmp_path / "alerts.db"
    store = AlertStore(str(db), dry_run=True)
    assert len(store.check_and_update_batch([anomaly()], 3)) == 1
    assert not db.exists()


def test_dry_run_is_repeatable(tmp_path):
    db = str(tmp_path / "alerts.db")
    for _ in range(3):
        assert len(AlertStore(db, dry_run=True).check_and_update_batch([anomaly()], 3)) == 1


def test_dry_run_sees_history_but_does_not_modify_it(tmp_path):
    db = str(tmp_path / "alerts.db")
    real = AlertStore(db)
    real.check_and_update_batch([anomaly(18)], 3)
    real.mark_email_sent([FP])
    before = rows(db)

    dry = AlertStore(db, dry_run=True)
    assert dry.check_and_update_batch([anomaly(19)], 3) == []  # inside cooldown, like a real run
    assert len(dry.check_and_update_batch([anomaly(25)], 3)) == 1  # outside cooldown
    assert rows(db) == before


def test_severity_escalation_bypasses_cooldown():
    store = AlertStore(":memory:")
    store.check_and_update_batch([anomaly(18, "medium")], 3)
    store.mark_email_sent([FP])
    assert len(store.check_and_update_batch([anomaly(19, "high")], 3)) == 1


def test_narration_and_sent_flag_roundtrip(tmp_path):
    db = str(tmp_path / "alerts.db")
    store = AlertStore(db)
    store.check_and_update_batch([anomaly()], 3)
    fp = store.get_fingerprint("sales", "total", "up")
    store.save_narration(fp, "hello")
    store.mark_email_sent([fp])
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT summary_text, email_sent_flag FROM alerts").fetchone() == ("hello", 1)
    finally:
        conn.close()
