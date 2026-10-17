import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import time

class AlertStore:
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
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
                    email_sent_flag INTEGER
                )
            """)

    def _get_connection(self):
        # Add timeout to handle locking
        conn = sqlite3.connect(self.db_path, timeout=30)
        return conn

    def get_fingerprint(self, metric: str, segment: str, direction: str) -> str:
        s = f"{metric}:{segment}:{direction}"
        return hashlib.md5(s.encode()).hexdigest()

    def check_and_update_batch(self, anomalies: List[Dict[str, Any]], cooldown_days: int) -> List[Dict[str, Any]]:
        to_alert = []
        now = datetime.now()
        
        with self._get_connection() as conn:
            for anomaly in anomalies:
                fingerprint = self.get_fingerprint(anomaly["metric"], anomaly["segment"], anomaly["direction"])
                
                cursor = conn.execute("SELECT last_seen, severity FROM alerts WHERE fingerprint = ?", (fingerprint,))
                row = cursor.fetchone()
                
                if row:
                    last_seen, old_severity = row
                    last_seen_dt = datetime.fromisoformat(last_seen)
                    days_since = (now - last_seen_dt).days
                    
                    if days_since < cooldown_days and anomaly["severity"] == old_severity:
                        continue # Suppress
                    
                    conn.execute("""
                        UPDATE alerts SET last_seen = ?, severity = ?, email_sent_flag = 0
                        WHERE fingerprint = ?
                    """, (now.isoformat(), anomaly["severity"], fingerprint))
                else:
                    conn.execute("""
                        INSERT INTO alerts (fingerprint, metric, segment, direction, first_seen, last_seen, severity, status, email_sent_flag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (fingerprint, anomaly["metric"], anomaly["segment"], anomaly["direction"], 
                          now.isoformat(), now.isoformat(), anomaly["severity"], "open", 0))
                
                to_alert.append(anomaly)
        
        return to_alert

    def mark_email_sent(self, fingerprints: List[str]):
        with self._get_connection() as conn:
            for fp in fingerprints:
                conn.execute("UPDATE alerts SET email_sent_flag = 1 WHERE fingerprint = ?", (fp,))
