from src.anomaly_agent.config import load_config
from src.anomaly_agent.detection import Detector
from src.anomaly_agent.ingestion import Ingestion
from src.anomaly_agent.metrics import MetricEngine

from .inject import AnomalyInjector


def run_evaluation(config_path: str):
    config = load_config(config_path)
    ingestion = Ingestion(config)
    df = ingestion.load_data()
    df, _ = ingestion.validate_and_clean(df)

    engine = MetricEngine(config)
    series_dict = engine.process(df)

    # Take a clean series
    series = series_dict[("sales", "total")].copy()

    injector = AnomalyInjector()

    # Inject a spike
    target_date = series.index[len(series) // 2]
    injected_series = injector.inject(series, "spike", target_date, 0.5)

    # Run detection
    detector = Detector(config)
    # We need to wrap the series back into the dict format
    test_dict = {("sales", "total"): injected_series}
    anomalies = detector.detect(test_dict)

    print(f"Evaluation Results for Spike at {target_date}:")
    found = any(a["date"] == target_date for a in anomalies)
    print(f"  Detected injected anomaly: {found}")
    if found:
        anomaly = next(a for a in anomalies if a["date"] == target_date)
        print(f"  Z-Score: {anomaly['z_score']:.2f}")
        print(f"  Relative Deviation: {anomaly['rel_deviation']:.2%}")


if __name__ == "__main__":
    run_evaluation("config/dataco_supply_chain.yaml")
