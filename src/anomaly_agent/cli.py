import argparse
import logging
import smtplib
import sys
import traceback
from typing import Any

import pandas as pd

from .alerting.email import EmailAlerter
from .attribution import AttributionEngine
from .config import load_config
from .detection import Detector
from .forecast_context import add_forecast_context
from .ingestion import Ingestion
from .metrics import MetricEngine
from .narration.client import Narrator, build_client
from .severity import SeverityScorer
from .store import AlertStore
from .variance_context import add_variance_context

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
TOP_N = 5

logger = logging.getLogger(__name__)


def attach_drivers(
    headline: list[dict[str, Any]],
    segment_anomalies: list[dict[str, Any]],
    max_drivers: int = 3,
) -> None:
    """List recently abnormal segments as possible drivers of a headline anomaly:
    same metric and direction, segment period ending within the 14 days before (or on)
    the headline date. Weekly segment series only cover completed weeks, so the match
    is deliberately a recent-window match, not an exact-day match."""
    for h in headline:
        matches = []
        for s in segment_anomalies:
            if s["metric"] != h["metric"] or s["direction"] != h["direction"]:
                continue
            lag_days = (h["date"] - s["date"]).days
            if 0 <= lag_days <= 14 or (-s["period_days"] < lag_days < 0):
                matches.append(s)
        matches.sort(key=lambda s: s["abs_deviation"], reverse=True)
        h["drivers"] = [f"{s['segment']} ({s['rel_deviation']:.0%} vs expected)" for s in matches[:max_drivers]]


def add_attribution(anomalies: list[dict[str, Any]], df: pd.DataFrame, config: Any) -> None:
    """Attach segment drivers (contribution, or mix/rate for ratios) to headline anomalies.

    A failure on one anomaly is logged and skipped; it never stops the run."""
    engine = AttributionEngine(config)
    metric_types = {m.name: m.type for m in config.metrics}
    for a in anomalies:
        a["metric_type"] = metric_types.get(a["metric"])
        if a["segment"] != "total":
            continue
        try:
            result = engine.attribute(a, df)
        except Exception as e:
            logger.warning("Attribution failed for %s: %s: %s", a["metric"], type(e).__name__, e)
            continue
        if not result:
            continue
        a["attribution"] = result
        if result["drivers"]:
            a["drivers"] = result["drivers"]
        if result["note"]:
            a["driver_note"] = result["note"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Business Metrics Anomaly Agent")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Do not send emails or write to the alert store")
    parser.add_argument("--as-of", help="Treat data after this date (YYYY-MM-DD) as unseen")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Scan the whole history instead of only the latest periods (experiments)",
    )
    args = parser.parse_args()

    try:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)

        print("Initializing Ingestion...")
        ingestion = Ingestion(config)
        df = ingestion.load_data()
        df, _quality_report = ingestion.validate_and_clean(df)

        as_of = pd.Timestamp(args.as_of) if args.as_of else None
        if as_of is not None:
            date_col = config.dataset.date_column
            df = df[df[date_col] < as_of + pd.Timedelta(days=1)]
        print(
            f"Data: {len(df):,} rows, {df[config.dataset.date_column].min().date()} "
            f"to {df[config.dataset.date_column].max().date()}"
        )

        print("Initializing Metric Engine...")
        series_dict = MetricEngine(config).process(df)

        print("Initializing Detection...")
        window = None if args.full_history else config.alerting.alert_window
        anomalies = Detector(config).detect(series_dict, as_of=as_of, window=window)
        # Forward-looking context is separate from the detector baseline for past anomalies.
        add_forecast_context(anomalies, series_dict, config)
        print(
            f"Detected {len(anomalies)} anomalies ({'full history' if window is None else f'last {window} periods'})."
        )

        if config.alerting.headline_only:
            segment_anoms = [a for a in anomalies if a["segment"] != "total"]
            anomalies = [a for a in anomalies if a["segment"] == "total"]
            attach_drivers(anomalies, segment_anoms)
            print(f"Headline anomalies: {len(anomalies)} (segment anomalies used as drivers: {len(segment_anoms)})")

        if not anomalies:
            print("No anomalies detected.")
            return

        add_attribution(anomalies, df, config)
        # Add a reconciled volume-price-mix explanation when source fields permit it.
        add_variance_context(anomalies, df, config)

        scorer = SeverityScorer(config)
        min_level = SEVERITY_RANK.get(config.alerting.min_severity.lower(), 1)
        scored = []
        for a in anomalies:
            a["severity"] = scorer.score(a)
            if SEVERITY_RANK.get(a["severity"].lower(), 0) >= min_level:
                scored.append(a)
        counts = pd.Series([a["severity"] for a in scored]).value_counts().to_dict()
        print(f"Met min_severity ({config.alerting.min_severity}): {len(scored)} {counts}")
        if not scored:
            return

        store = AlertStore(dry_run=args.dry_run)
        to_alert = store.check_and_update_batch(scored, config.alerting.cooldown_days)
        print(f"{len(to_alert)} alerts after deduplication ({len(scored) - len(to_alert)} suppressed or merged).")
        if not to_alert:
            print("All anomalies suppressed by cooldown.")
            return

        to_alert = sorted(
            to_alert,
            key=lambda x: (SEVERITY_RANK.get(x["severity"], 0), abs(x["z_score"])),
            reverse=True,
        )[:TOP_N]

        print(f"Generating Narrations (top {len(to_alert)})...")
        narrator = Narrator(build_client(config), max_retries=config.llm.max_retries)
        for a in to_alert:
            a["narration"] = narrator.narrate(a)
            store.save_narration(
                store.get_fingerprint(a["metric"], a["segment"], a["direction"]),
                a["narration"],
            )
        print(f"Narrations: {narrator.narrated} from LLM, {narrator.fallbacks} fallbacks")

        if args.dry_run:
            for a in to_alert:
                print(f"\n[{a['severity'].upper()}] {a['metric']} / {a['segment']} on {a['date'].date()}")
                print(a["narration"])
            print()

        print("Sending Alerts...")
        try:
            EmailAlerter(config).send_digest(to_alert, dry_run=args.dry_run)
        except (smtplib.SMTPException, OSError) as e:
            print(f"Email delivery failed ({type(e).__name__}: {e}).")
            print("The alerts are stored as pending and will be retried on the next run.")
            sys.exit(1)

        if not args.dry_run:
            store.mark_email_sent([store.get_fingerprint(a["metric"], a["segment"], a["direction"]) for a in to_alert])

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
