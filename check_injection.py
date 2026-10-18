"""Injection check on the real DataCo data: does the detector behave as claimed?"""
import pandas as pd

from anomaly_agent.config import load_config
from anomaly_agent.detection import Detector
from anomaly_agent.ingestion import Ingestion
from anomaly_agent.metrics import MetricEngine

cfg = load_config("config/dataco_supply_chain.yaml")
base, _ = Ingestion(cfg).validate_and_clean(Ingestion(cfg).load_data())
det = Detector(cfg)


def run(df, as_of):
    sd = MetricEngine(cfg).process(df)
    head = {k: v for k, v in sd.items() if k[1] == "total"}
    found = det.detect(head, as_of=pd.Timestamp(as_of), window=3)
    return [
        (a["date"].date().isoformat(), a["direction"], round(a["z_score"], 1))
        for a in found
        if a["metric"] == "sales"
    ]


def inject(days, factor):
    df = base.copy()
    mask = df.order_date.dt.normalize().isin(pd.to_datetime(days))
    df.loc[mask, "sales"] *= factor
    return df


cases = [
    ("clean data                 ", base, "2016-06-17", "expect: []"),
    ("3x spike on 2016-06-15     ", inject(["2016-06-15"], 3), "2016-06-17", "expect: up alert dated 2016-06-15"),
    ("1.35x one-day blip 03-10   ", inject(["2016-03-10"], 1.35), "2016-03-12", "expect: []"),
    ("1.5x for 3 days 09-05..07  ", inject(["2016-09-05", "2016-09-06", "2016-09-07"], 1.5), "2016-09-08", "expect: up alert around 2016-09-07"),
]
for name, df, as_of, expect in cases:
    print(f"{name}: {run(df, as_of)}   <- {expect}")
