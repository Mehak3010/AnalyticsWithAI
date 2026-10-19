"""Run the evaluation harness against the real pipeline.

python -m evaluation.run_eval --config config/dataco_supply_chain.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_agent.attribution import AttributionEngine
from anomaly_agent.config import load_config
from anomaly_agent.detection import Detector
from anomaly_agent.ingestion import Ingestion
from anomaly_agent.metrics import MetricEngine

from .baselines import fixed_threshold_flags, moving_average_z_flags, replay_flags
from .harness import EvalSettings, make_replay_runner, natural_noise, run_series_eval
from .inject import inject_segment
from .report import write_outputs
from .scoring import score_attribution

logger = logging.getLogger(__name__)


def headline_series(config: Any, df: pd.DataFrame, metric_cfg: Any) -> pd.Series:
    """The daily total series for one metric, built the same way the pipeline builds it."""
    engine = MetricEngine(config)
    date_col = config.dataset.date_column
    engine._first = df[date_col].min().normalize()
    engine._last = df[date_col].max().normalize()
    return engine._compute_series(df, metric_cfg, date_col, config.dataset.grain)


def make_detect_fn(config: Any, metric: str):
    detector = Detector(config)
    window = config.alerting.alert_window
    key = (metric, "total")

    def detect(series: pd.Series, as_of: pd.Timestamp):
        try:
            found = detector.detect({key: series}, as_of=as_of, window=window)
        except ValueError as e:  # not enough history for this as-of date
            logger.debug("detect skipped for %s: %s", as_of.date(), e)
            return []
        return [
            (pd.Timestamp(a["date"]).normalize(), a["direction"])
            for a in found
            if a["metric"] == metric and a["segment"] == "total"
        ]

    return detect


def run_attribution_eval(
    config: Any,
    df: pd.DataFrame,
    metric_cfg: Any,
    dates: list[pd.Timestamp],
    n_cases: int,
    total_move: float,
    seed: int,
) -> dict[str, Any]:
    """Scale one segment's rows on one day, then check the engine names that segment.

    Attribution only runs after a real alert, so each injection is sized to move the *total* by
    ``total_move`` (e.g. 30%), not by a fixed percentage of the segment.
    """
    if metric_cfg.type != "additive":
        return {"skipped": "attribution accuracy is measured on additive metrics"}
    date_col, column = config.dataset.date_column, metric_cfg.column
    pairs = []
    for dim in config.dataset.dimensions:
        if dim not in df.columns:
            continue
        shares = df.groupby(dim, observed=True)[column].sum()
        shares = shares / shares.sum()
        pairs += [(dim, value) for value, share in shares.items() if 0.10 <= share <= 0.60]
    if not pairs or not dates:
        return {"skipped": "no segments with a 10-60% share, or no injection dates"}

    rng = np.random.default_rng(seed)
    engine = AttributionEngine(config)
    cases = []
    for i, pair_idx in enumerate(rng.permutation(len(pairs))[:n_cases]):
        dim, value = pairs[pair_idx]
        date = dates[i % len(dates)]
        window = df[(df[date_col] >= date - pd.Timedelta(days=40)) & (df[date_col] < date + pd.Timedelta(days=1))]
        day = window[window[date_col].dt.normalize() == date.normalize()]
        day_total = float(day[column].sum())
        share = float(day.loc[day[dim] == value, column].sum()) / day_total if day_total else 0.0
        if share < 0.05:
            continue
        injected, n_rows = inject_segment(window, date_col, column, dim, value, date, total_move / share)
        if n_rows == 0:
            continue
        result = engine.attribute({"metric": metric_cfg.name, "segment": "total", "date": date}, injected)
        in_dim, story = score_attribution(result, dim, value)
        cases.append(
            {"dimension": dim, "value": str(value), "date": str(date.date()), "in_dimension": in_dim, "story": story}
        )
    n = len(cases)
    return {
        "total_move": total_move,
        "n": n,
        "cases": cases,
        "top1_in_dimension": sum(c["in_dimension"] for c in cases) / n if n else None,
        "top1_story": sum(c["story"] for c in cases) / n if n else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate detection and attribution with injected anomalies")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metric", default="sales", help="additive headline metric to inject into")
    parser.add_argument("--dates", type=int, default=5, help="injection dates per type and size")
    parser.add_argument("--eval-days", type=int, default=240, help="length of the replay window")
    parser.add_argument("--attr-cases", type=int, default=12, help="attribution test cases (0 to skip)")
    parser.add_argument("--attr-total-move", type=float, default=0.30, help="size of the injected move in the total")
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.1, 0.2, 0.5], help="injected anomaly sizes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="evaluation", help="output folder for report.md and results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    ingestion = Ingestion(config)
    df, _ = ingestion.validate_and_clean(ingestion.load_data())
    metric_cfg = next(m for m in config.metrics if m.name == args.metric)
    series = headline_series(config, df, metric_cfg)

    settings = EvalSettings(magnitudes=tuple(args.sizes), n_dates=args.dates, eval_days=args.eval_days, seed=args.seed)
    runners = {
        "detector": make_replay_runner(make_detect_fn(config, args.metric)),
        "fixed_threshold": lambda s, dates: replay_flags(fixed_threshold_flags(s), dates),
        "moving_avg_z": lambda s, dates: replay_flags(moving_average_z_flags(s), dates),
    }
    n_cases = len(settings.types) * len(settings.magnitudes) * args.dates
    print(f"Running {n_cases} injected cases x 3 methods over {args.eval_days} days (this can take a few minutes)...")
    detection = run_series_eval(series, runners, settings)

    attribution: dict[str, Any] = {"skipped": "disabled"}
    if args.attr_cases > 0:
        print("Running attribution cases...")
        attribution = run_attribution_eval(
            config, df, metric_cfg, detection["dates"], args.attr_cases, args.attr_total_move, args.seed
        )

    payload = {
        "setup": {
            "metric": args.metric,
            "dataset": config.dataset.path,
            "seed": args.seed,
            "noise": natural_noise(series),
        },
        "detection": detection,
        "attribution": attribution,
    }
    paths = write_outputs(Path(args.out), payload)
    print(f"Wrote {paths['report']} and {paths['results']}")


if __name__ == "__main__":
    main()
