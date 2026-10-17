import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from ..config import Config
from dotenv import load_dotenv

load_dotenv()

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
            print(f"DRY RUN: Email would be sent to {self.config.alerting.email['to']}")
            print(f"Subject: {subject}")
            with open("reports/last_email_digest.html", "w") as f:
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

    def _build_html_body(self, anomalies: List[Dict[str, Any]]) -> str:
        rows = ""
        for a in sorted(anomalies, key=lambda x: x["severity"], reverse=True):
            rows += f"""
            <tr>
                <td><b>[{a['severity'].upper()}]</b> {a['metric']}</td>
                <td>{a['segment']}</td>
                <td>{a['observed']:.2f}</td>
                <td>{a['expected']:.2f}</td>
                <td>{a['rel_deviation']:.1%}</td>
            </tr>
            <tr>
                <td colspan="5"><i>{a.get('narration', 'No narration available.')}</i></td>
            </tr>
            <tr><td colspan="5"><hr></td></tr>
            """
            
        html = f"""
        <html>
        <body>
            <h2>Anomaly Digest</h2>
            <table width="100%">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Segment</th>
                        <th>Observed</th>
                        <th>Expected</th>
                        <th>Deviation</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </body>
        </html>
        """
        return html
