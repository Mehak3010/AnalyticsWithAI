"""Pure scoring logic for the evaluation harness (no pipeline imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Flag = tuple[pd.Timestamp, str]  # (anomaly date, direction)


@dataclass(frozen=True)
class Case:
    anomaly_type: str
    magnitude: float
    date: pd.Timestamp
    direction: str


@dataclass(frozen=True)
class CaseResult:
    case: Case
    detected: bool
    delay_days: int | None
    tp: int  # distinct alerts that match the injected anomaly
    fp: int  # distinct alerts that do not (and were not already present on clean data)


def is_match(case: Case, anomaly_date: pd.Timestamp, direction: str, max_delay: int) -> bool:
    """A flag counts if its direction is right and it lands where the injection can show up.

    Spikes and drops affect one day, so the flag must be on that day. Shifts and drifts persist,
    so a flag on any of the first ``max_delay`` days after the start counts.
    """
    if direction != case.direction:
        return False
    offset = (pd.Timestamp(anomaly_date) - case.date).days
    if case.anomaly_type in ("spike", "drop"):
        return offset == 0
    return 0 <= offset <= max_delay


def score_case(
    case: Case,
    flagged_by_asof: dict[pd.Timestamp, list[Flag]],
    clean_flags: set[Flag],
    max_delay: int,
) -> CaseResult:
    """Score one injected case from a day-by-day replay.

    Flags that also appear on the clean series are ignored (they are not caused by the injection).
    Detection delay = days from the injection to the first as-of date that reports a matching flag.
    """
    matched: set[Flag] = set()
    extras: set[Flag] = set()
    delay: int | None = None
    for as_of in sorted(flagged_by_asof):
        for flag in flagged_by_asof[as_of]:
            date, direction = flag
            if is_match(case, date, direction, max_delay):
                matched.add(flag)
                if delay is None:
                    delay = (as_of - case.date).days
            elif flag not in clean_flags:
                extras.add(flag)
    return CaseResult(case, detected=bool(matched), delay_days=delay, tp=len(matched), fp=len(extras))


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Recall is per case; precision is per alert. F1 combines the two."""
    n = len(results)
    detected = [r for r in results if r.detected]
    tp, fp = sum(r.tp for r in results), sum(r.fp for r in results)
    recall = len(detected) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = 0.0 if n else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    delays = [r.delay_days for r in detected if r.delay_days is not None]
    return {
        "cases": n,
        "detected": len(detected),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "median_delay": float(np.median(delays)) if delays else None,
        "true_alerts": tp,
        "false_alerts": fp,
    }


def false_alarms_per_100_days(clean_flags: set[Flag], n_days: int) -> float:
    return len(clean_flags) / n_days * 100 if n_days else 0.0


def score_attribution(result: dict[str, Any], dim: str, value: object) -> tuple[bool, bool]:
    """(top contributor within the injected dimension is right, the headline story is right)."""
    target = f"{dim}={value}"
    in_dim_list = result.get("by_dimension", {}).get(dim, {}).get("contributors") or []
    story_list = result.get("contributors") or []
    return (
        bool(in_dim_list) and in_dim_list[0]["segment"] == target,
        bool(story_list) and story_list[0]["segment"] == target,
    )
