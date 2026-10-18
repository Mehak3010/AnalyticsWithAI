import hashlib
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value))


class AlertStore:
    """SQLite alert log. first_seen / last_seen hold the anomaly's DATA date;
    created_at holds the wall-clock time the alert was first stored."""

    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    fingerprint TEXT PRIMARY KEY,
                    metric TEXT,
                    segment TEXT,
                    direction TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    severity TEXT,
                    status TEXT,
                    summary_text TEXT,
                    email_sent_flag INTEGER,
                    created_at TEXT
                )
                """
            )
            # Older databases lack created_at
            cols = [r[1] for r in conn.execute("PRAGMA table_info(alerts)")]
            if "created_at" not in cols:
                conn.execute("ALTER TABLE alerts ADD COLUMN created_at TEXT")

    def get_fingerprint(self, metric: str, segment: str, direction: str) -> str:
        return hashlib.md5(f"{metric}:{segment}:{direction}".encode()).hexdigest()

    def check_and_update_batch(
        self, anomalies: List[Dict[str, Any]], cooldown_days: int
    ) -> List[Dict[str, Any]]:
        """Return anomalies that should alert. Suppress a repeat of the same
        metric/segment/direction inside the cooldown window unless severity escalated."""
        now = datetime.now()

        # Several anomalies can share a fingerprint in one run: keep the latest, most severe
        best: Dict[str, Dict[str, Any]] = {}
        for a in anomalies:
            fp = self.get_fingerprint(a["metric"], a["segment"], a["direction"])
            cur = best.get(fp)
            key = (_to_datetime(a.get("date")) or now, SEVERITY_RANK.get(a["severity"], 0))
            if cur is None or key > (
                _to_datetime(cur.get("date")) or now,
                SEVERITY_RANK.get(cur["severity"], 0),
            ):
                best[fp] = a

        to_alert: List[Dict[str, Any]] = []
        with self._get_connection() as conn:
            for fp, a in best.items():
                a_date = _to_datetime(a.get("date")) or now
                row = conn.execute(
                    "SELECT last_seen, severity FROM alerts WHERE fingerprint = ?", (fp,)
                ).fetchone()

                if row:
                    last_date, old_sev = _to_datetime(row[0]), row[1]
                    escalated = SEVERITY_RANK.get(a["severity"], 0) > SEVERITY_RANK.get(old_sev, 0)
                    days = (a_date - last_date).days
                    if (days < cooldown_days) and not escalated:
                        continue  # repeat inside cooldown (or older than what we have)
                    conn.execute(
                        "UPDATE alerts SET last_seen = ?, severity = ?, email_sent_flag = 0 "
                        "WHERE fingerprint = ?",
                        (a_date.isoformat(), a["severity"], fp),
                    )
                else:
                    conn.execute(
                        "INSERT INTO alerts (fingerprint, metric, segment, direction, first_seen, "
                        "last_seen, severity, status, email_sent_flag, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', 0, ?)",
                        (fp, a["metric"], a["segment"], a["direction"], a_date.isoformat(),
                         a_date.isoformat(), a["severity"], now.isoformat()),
                    )
                to_alert.append(a)
        return to_alert

    def save_narration(self, fingerprint: str, text: str):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE alerts SET summary_text = ? WHERE fingerprint = ?", (text, fingerprint)
            )

    def mark_email_sent(self, fingerprints: List[str]):
        with self._get_connection() as conn:
            for fp in fingerprints:
                conn.execute("UPDATE alerts SET email_sent_flag = 1 WHERE fingerprint = ?", (fp,))
