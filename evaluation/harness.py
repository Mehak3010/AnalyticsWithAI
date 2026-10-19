"""Replay-based evaluation of detection methods on a series with injected anomalies.

A *runner* is any function ``(series, as_of_dates) -> {as_of: [(anomaly_date, direction), ...]}``.
The real detector, and the two naive baselines, are all wrapped as runners.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inject import SERIES_TYPES, AnomalyInjector, injected_direction
from .scoring import Case, CaseResult, Flag, false_alarms_per_100_days, score_case

Replay = dict[pd.Timestamp, list[Flag]]
Runner = Callable[[pd.Series, list[pd.Timestamp]], Replay]
DetectFn = Callable[[pd.Series, pd.Timestamp], list[Flag]]


@dataclass
class EvalSettings:
    types: tuple[str, ...] = SERIES_TYPES
    magnitudes: tuple[float, ...] = (0.1, 0.2, 0.5)
    n_dates: int = 5
    max_delay: int = 7
    eval_days: int = 240
    min_gap_days: int = 14
    seed: int = 42


def natural_noise(series: pd.Series, weeks: int = 4) -> float | None:
    """Robust day-to-day noise: spread of each day's deviation from its same-weekday median (as a fraction)."""
    ref = pd.concat([series.shift(7 * k) for k in range(1, weeks + 1)], axis=1).median(axis=1)
    dev = (series / ref - 1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dev) < 10:
        return None
    return float(1.4826 * np.median(np.abs(dev - dev.median())))


def make_replay_runner(detect_fn: DetectFn) -> Runner:
    """Wrap a detector: on each as-of date it only sees data up to that date, like a scheduled run."""

    def runner(series: pd.Series, as_of_dates: list[pd.Timestamp]) -> Replay:
        return {d: detect_fn(series[series.index <= d], d) for d in as_of_dates}

    return runner


def sample_dates(
    candidates: list[pd.Timestamp],
    avoid: list[pd.Timestamp],
    n: int,
    max_delay: int,
    min_gap: int,
    rng: np.random.Generator,
) -> list[pd.Timestamp]:
    """Pick injection dates away from natural anomalies and from each other."""
    ok = [d for d in candidates if all(abs((d - a).days) > max_delay for a in avoid)]
    chosen: list[pd.Timestamp] = []
    for i in rng.permutation(len(ok)):
        d = ok[i]
        if all(abs((d - c).days) >= min_gap for c in chosen):
            chosen.append(d)
        if len(chosen) == n:
            break
    return sorted(chosen)


def run_series_eval(series: pd.Series, runners: dict[str, Runner], settings: EvalSettings) -> dict:
    """Inject every (type, magnitude) at several dates and score every runner on the same cases."""
    series = series.astype(float)
    end = pd.Timestamp(series.index.max())
    start = end - pd.Timedelta(days=settings.eval_days)
    # Only use timestamps that exist in the input series. Injecting on a calendar day absent
    # from a sparse series is a no-op, which silently turns a labelled anomaly into a missed case.
    eval_dates = [pd.Timestamp(d) for d in series.index if start <= pd.Timestamp(d) <= end]
    rng = np.random.default_rng(settings.seed)

    # Clean replay: false alarms, and the natural anomalies that must not count against the injection.
    clean_flags: dict[str, set[Flag]] = {}
    for name, runner in runners.items():
        replay = runner(series, eval_dates)
        clean_flags[name] = {f for flags in replay.values() for f in flags}

    primary = next(iter(runners))
    candidates = [d for d in eval_dates if d + pd.Timedelta(days=settings.max_delay) <= end]
    avoid = [d for d, _ in clean_flags[primary]]
    dates = sample_dates(candidates, avoid, settings.n_dates, settings.max_delay, settings.min_gap_days, rng)
    if not dates:
        raise ValueError(
            f"No usable injection dates: the {primary} flagged {len(clean_flags[primary])} natural anomalies "
            f"in the {len(eval_dates)}-day replay window, leaving no clean stretch. "
            "Increase eval_days or review the thresholds."
        )

    injector = AnomalyInjector(settings.seed)
    results: dict[str, list[CaseResult]] = {name: [] for name in runners}
    for anomaly_type in settings.types:
        for magnitude in settings.magnitudes:
            for date in dates:
                injected = injector.inject(series, anomaly_type, date, magnitude)
                case = Case(anomaly_type, magnitude, date, injected_direction(anomaly_type, magnitude))
                window = [d for d in eval_dates if date <= d <= date + pd.Timedelta(days=settings.max_delay)]
                for name, runner in runners.items():
                    replay = runner(injected, window)
                    results[name].append(score_case(case, replay, clean_flags[name], settings.max_delay))

    return {
        "results": results,
        "false_alarms_per_100d": {n: false_alarms_per_100_days(f, len(eval_dates)) for n, f in clean_flags.items()},
        "clean_alert_counts": {n: len(f) for n, f in clean_flags.items()},
        "n_clean_days": len(eval_dates),
        "dates": dates,
        "eval_start": eval_dates[0],
        "eval_end": end,
    }
