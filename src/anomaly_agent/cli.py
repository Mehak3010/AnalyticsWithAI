import argparse
import sys
from .config import load_config
from .ingestion import Ingestion
from .metrics import MetricEngine
from .detection import Detector
from .severity import SeverityScorer
from .narration.client import LLMClient
from .store import AlertStore
from .alerting.email import EmailAlerter

def main():
    parser = argparse.ArgumentParser(description="Business Metrics Anomaly Agent")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Do not send emails")
    args = parser.parse_args()
    
    try:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
        
        print("Initializing Ingestion...")
        ingestion = Ingestion(config)
        df = ingestion.load_data()
        df, quality_report = ingestion.validate_and_clean(df)
            
        print("Initializing Metric Engine...")
        engine = MetricEngine(config)
        series_dict = engine.process(df)
        
        print("Initializing Detection...")
        detector = Detector(config)
        anomalies = detector.detect(series_dict)
        print(f"Detected {len(anomalies)} anomalies.")
        
        if not anomalies:
            print("No anomalies detected.")
            return

        print("Scoring Severity and Deduplicating...")
        scorer = SeverityScorer(config)
        store = AlertStore()
        
        for anomaly in anomalies:
            anomaly["severity"] = scorer.score(anomaly)
            
        to_alert = store.check_and_update_batch(anomalies, config.alerting.cooldown_days)
        
        print(f"{len(to_alert)} anomalies remaining after deduplication.")
        
        if not to_alert:
            print("All anomalies suppressed by cooldown.")
            return

        print("Generating Narrations (Top 5)...")
        llm = LLMClient(config)
        to_alert = sorted(to_alert, key=lambda x: x["severity"], reverse=True)[:5]
        for anomaly in to_alert:
            anomaly["narration"] = llm.narrate(anomaly)
            
        print("Sending Alerts...")
        alerter = EmailAlerter(config)
        alerter.send_digest(to_alert, dry_run=args.dry_run)
        
        if not args.dry_run:
            store.mark_email_sent([store.get_fingerprint(a["metric"], a["segment"], a["direction"]) for a in to_alert])

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
