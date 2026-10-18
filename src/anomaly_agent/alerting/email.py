import html
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

from dotenv import load_dotenv

from ..config import Config

load_dotenv()

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class EmailAlerter:
    def __init__(self, config: Config):
        self.config = config
        self.host = os.getenv("MAILTRAP_HOST")
        self.port = int(os.getenv("MAILTRAP_PORT", 587))
        self.username = os.getenv("MAILTRAP_USERNAME")
        self.password = os.getenv("MAILTRAP_PASSWORD")

    def send_digest(self, anomalies: List[Dict[str, Any]], dry_run: bool = False):
        if not anomalies:
            print("No anomalies to send.")
            return

        subject = f"{self.config.alerting.email['subject_prefix']} Anomaly Digest - {len(anomalies)} alerts"
        body = self._build_html_body(anomalies)

        if dry_run:
            os.makedirs("reports", exist_ok=True)
            print(f"DRY RUN: Email would be sent to {self.config.alerting.email['to']}")
            print(f"Subject: {subject}")
            with open("reports/last_email_digest.html", "w", encoding="utf-8") as f:
                f.write(body)
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = "alerts@anomaly-agent.ai"
        msg["To"] = ", ".join(self.config.alerting.email["to"])
        msg.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            print("Email digest sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
            raise  # let the caller know, so alerts are not marked as sent

    def _build_html_body(self, anomalies: List[Dict[str, Any]]) -> str:
        ordered = sorted(
            anomalies,
            key=lambda x: (SEVERITY_RANK.get(x["severity"], 0), abs(x.get("z_score", 0))),
            reverse=True,
        )
        rows = ""
        for a in ordered:
            date = a.get("date")
            date_txt = date.date().isoformat() if hasattr(date, "date") else str(date or "")
            drivers = a.get("drivers") or []
            drivers_txt = "Possible drivers: " + "; ".join(drivers) if drivers else ""
            narration = html.escape(a.get("narration") or "No narration available.")
            rows += f"""
            <tr>
                <td><b>[{a['severity'].upper()}]</b> {html.escape(a['metric'])}</td>
                <td>{html.escape(a['segment'])}</td>
                <td>{date_txt}</td>
                <td>{a['observed']:.2f}</td>
                <td>{a['expected']:.2f}</td>
                <td>{a['rel_deviation']:.1%}</td>
            </tr>
            <td colspan="6" style="white-space: pre-line"><i>{narration}</i></td>
            {f'<tr><td colspan="6">{html.escape(drivers_txt)}</td></tr>' if drivers_txt else ''}
            <tr><td colspan="6"><hr></td></tr>
            """

        return f"""
        <html>
        <body>
            <h2>Anomaly Digest</h2>
            <table width="100%">
                <thead>
                    <tr>
                        <th>Metric</th><th>Segment</th><th>Date</th>
                        <th>Observed</th><th>Expected</th><th>Deviation</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </body>
        </html>
        """
