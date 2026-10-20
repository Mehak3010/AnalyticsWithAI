import html
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from dotenv import load_dotenv

from ..config import Config

load_dotenv()

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

SEVERITY_COLORS = {
    "low": ("#166534", "#dcfce7"),
    "medium": ("#92400e", "#fef3c7"),
    "high": ("#9a3412", "#ffedd5"),
    "critical": ("#991b1b", "#fee2e2"),
}


def escape(value: Any) -> str:
    """Safely escape dynamic content before inserting it into HTML."""
    return html.escape(str(value if value is not None else ""))


class EmailAlerter:
    def __init__(self, config: Config):
        self.config = config
        self.host = os.getenv("MAILTRAP_HOST")
        self.port = int(os.getenv("MAILTRAP_PORT", "587"))
        self.username = os.getenv("MAILTRAP_USERNAME")
        self.password = os.getenv("MAILTRAP_PASSWORD")

    def send_digest(self, anomalies: list[dict[str, Any]], dry_run: bool = False):
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
            raise

    def _build_html_body(self, anomalies: list[dict[str, Any]]) -> str:
        ordered = sorted(
            anomalies,
            key=lambda x: (
                SEVERITY_RANK.get(x["severity"], 0),
                abs(x.get("z_score", 0)),
            ),
            reverse=True,
        )

        cards = []

        for anomaly in ordered:
            severity = str(anomaly["severity"]).lower()
            severity_color, severity_bg = SEVERITY_COLORS.get(severity, ("#374151", "#f3f4f6"))

            date = anomaly.get("date")
            date_text = date.date().isoformat() if hasattr(date, "date") else str(date or "")

            metric = escape(anomaly["metric"])
            segment = escape(anomaly["segment"])
            narration = escape(anomaly.get("narration") or "No narration available.")
            severity_label = escape(severity.upper())

            observed = float(anomaly["observed"])
            expected = float(anomaly["expected"])
            deviation = float(anomaly["rel_deviation"])

            drivers = anomaly.get("drivers") or []
            driver_items = "".join(
                f"""
                <tr>
                    <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;
                               color:#374151;font-size:13px;">
                        {escape(driver)}
                    </td>
                </tr>
                """
                for driver in drivers
            )

            driver_section = ""
            if driver_items:
                driver_section = f"""
                <tr>
                    <td style="padding:18px 24px 6px;">
                        <div style="font-size:12px;font-weight:bold;
                                    letter-spacing:1px;color:#64748b;
                                    text-transform:uppercase;">
                            Possible drivers
                        </div>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 24px 8px;">
                        <table role="presentation" width="100%" cellspacing="0"
                               cellpadding="0"
                               style="border:1px solid #e5e7eb;
                                      border-radius:8px;">
                            {driver_items}
                        </table>
                    </td>
                </tr>
                """

            forecast_section = ""
            forecast = anomaly.get("forecast_context") or {}

            if forecast:
                forecast_model = escape(forecast.get("model", "Forecast"))
                forecast_date = escape(forecast.get("date", ""))
                forecast_value = float(forecast["value"])
                forecast_lower = float(forecast["lower"])
                forecast_upper = float(forecast["upper"])

                forecast_section = f"""
                <tr>
                    <td style="padding:16px 24px 22px;">
                        <table role="presentation" width="100%" cellspacing="0"
                               cellpadding="0"
                               style="background:#eff6ff;border:1px solid #bfdbfe;
                                      border-radius:8px;">
                            <tr>
                                <td style="padding:14px 16px;">
                                    <div style="font-size:11px;font-weight:bold;
                                                letter-spacing:1px;color:#1d4ed8;
                                                text-transform:uppercase;">
                                        Forward outlook · {forecast_model}
                                    </div>
                                    <div style="font-size:23px;font-weight:bold;
                                                color:#1e3a8a;padding-top:5px;">
                                        {forecast_value:,.2f}
                                    </div>
                                    <div style="font-size:12px;color:#475569;
                                                padding-top:5px;">
                                        Forecast date: {forecast_date}
                                    </div>
                                    <div style="font-size:12px;color:#475569;
                                                padding-top:3px;">
                                        Prediction interval:
                                        {forecast_lower:,.2f} –
                                        {forecast_upper:,.2f}
                                    </div>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                """

            cards.append(
                f"""
                <tr>
                    <td style="padding:0 0 22px;">
                        <table role="presentation" width="100%" cellspacing="0"
                               cellpadding="0"
                               style="background:#ffffff;border:1px solid #e2e8f0;
                                      border-radius:12px;overflow:hidden;">
                            <tr>
                                <td style="padding:20px 24px 16px;">
                                    <table role="presentation" width="100%"
                                           cellspacing="0" cellpadding="0">
                                        <tr>
                                            <td style="vertical-align:top;">
                                                <div style="font-size:11px;
                                                            font-weight:bold;
                                                            color:#64748b;
                                                            letter-spacing:1.2px;
                                                            text-transform:uppercase;">
                                                    KPI anomaly
                                                </div>
                                                <div style="font-size:22px;
                                                            font-weight:bold;
                                                            color:#0f172a;
                                                            padding-top:5px;">
                                                    {metric}
                                                </div>
                                                <div style="font-size:13px;
                                                            color:#64748b;
                                                            padding-top:4px;">
                                                    Segment: {segment}
                                                </div>
                                            </td>
                                            <td align="right"
                                                style="vertical-align:top;
                                                       padding-left:8px;">
                                                <span style="display:inline-block;
                                                             background:{severity_bg};
                                                             color:{severity_color};
                                                             padding:7px 11px;
                                                             border-radius:20px;
                                                             font-size:11px;
                                                             font-weight:bold;
                                                             letter-spacing:.5px;">
                                                    {severity_label}
                                                </span>
                                            </td>
                                        </tr>
                                    </table>
                                    <div style="font-size:12px;color:#64748b;
                                                padding-top:14px;">
                                        Detected on {escape(date_text)}
                                    </div>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:0 24px 20px;">
                                    <table role="presentation" width="100%"
                                           cellspacing="0" cellpadding="0">
                                        <tr>
                                            <td width="33%" style="padding:12px 8px;
                                                background:#f8fafc;border:1px solid #e2e8f0;
                                                border-radius:8px;vertical-align:top;">
                                                <div style="font-size:11px;color:#64748b;">
                                                    OBSERVED
                                                </div>
                                                <div style="font-size:17px;font-weight:bold;
                                                            color:#0f172a;padding-top:5px;">
                                                    {observed:,.2f}
                                                </div>
                                            </td>
                                            <td width="4%"></td>
                                            <td width="33%" style="padding:12px 8px;
                                                background:#f8fafc;border:1px solid #e2e8f0;
                                                border-radius:8px;vertical-align:top;">
                                                <div style="font-size:11px;color:#64748b;">
                                                    EXPECTED
                                                </div>
                                                <div style="font-size:17px;font-weight:bold;
                                                            color:#0f172a;padding-top:5px;">
                                                    {expected:,.2f}
                                                </div>
                                            </td>
                                            <td width="4%"></td>
                                            <td width="26%" style="padding:12px 8px;
                                                background:{severity_bg};
                                                border:1px solid {severity_bg};
                                                border-radius:8px;vertical-align:top;">
                                                <div style="font-size:11px;
                                                            color:{severity_color};">
                                                    DEVIATION
                                                </div>
                                                <div style="font-size:17px;font-weight:bold;
                                                            color:{severity_color};
                                                            padding-top:5px;">
                                                    {deviation:+.1%}
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:18px 24px;
                                           background:#f8fafc;
                                           border-top:1px solid #e2e8f0;">
                                    <div style="font-size:11px;font-weight:bold;
                                                letter-spacing:1px;color:#64748b;
                                                text-transform:uppercase;
                                                padding-bottom:10px;">
                                        Analysis & recommended action
                                    </div>
                                    <div style="font-size:13px;line-height:1.8;
                                                color:#334155;
                                                white-space:pre-line;">
                                        {narration}
                                    </div>
                                </td>
                            </tr>
                            {driver_section}
                            {forecast_section}
                        </table>
                    </td>
                </tr>
                """
            )

        alert_count = len(ordered)
        alert_word = "alert" if alert_count == 1 else "alerts"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anomaly Digest</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
             font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           style="background:#f1f5f9;">
        <tr>
            <td align="center" style="padding:28px 12px;">
                <table role="presentation" width="100%" cellspacing="0"
                       cellpadding="0"
                       style="max-width:720px;width:100%;">
                    <tr>
                        <td style="background:#0f172a;padding:26px 28px;
                                   border-radius:12px 12px 0 0;">
                            <div style="font-size:11px;font-weight:bold;
                                        letter-spacing:2px;color:#93c5fd;
                                        text-transform:uppercase;">
                                Monitoring · Automated alert
                            </div>
                            <div style="font-size:27px;font-weight:bold;
                                        color:#ffffff;padding-top:8px;">
                                Anomaly Digest
                            </div>
                            <div style="font-size:13px;line-height:1.6;
                                        color:#cbd5e1;padding-top:7px;">
                                KPI deviations detected by your monitoring agent.
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#ffffff;padding:20px 24px;
                                   border-left:1px solid #e2e8f0;
                                   border-right:1px solid #e2e8f0;">
                            <table role="presentation" width="100%"
                                   cellspacing="0" cellpadding="0">
                                <tr>
                                    <td>
                                        <div style="font-size:11px;font-weight:bold;
                                                    color:#64748b;
                                                    letter-spacing:1px;
                                                    text-transform:uppercase;">
                                            Report summary
                                        </div>
                                        <div style="font-size:22px;font-weight:bold;
                                                    color:#0f172a;padding-top:5px;">
                                            {alert_count} {alert_word} detected
                                        </div>
                                    </td>
                                    <td align="right" style="font-size:12px;
                                        color:#64748b;vertical-align:middle;">
                                        Review recommended
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#f8fafc;padding:20px 14px 2px;
                                   border:1px solid #e2e8f0;
                                   border-top:0;">
                            <table role="presentation" width="100%"
                                   cellspacing="0" cellpadding="0">
                                {"".join(cards)}
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#0f172a;padding:18px 24px;
                                   border-radius:0 0 12px 12px;text-align:center;">
                            <div style="font-size:12px;color:#cbd5e1;">
                                KPI Anomaly Detection &amp; Alerting Agent
                            </div>
                            <div style="font-size:11px;color:#94a3b8;
                                        padding-top:5px;">
                                Automated analysis · Review findings before taking action
                            </div>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
