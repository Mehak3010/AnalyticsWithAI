"""Evaluation metrics for timestamped anomaly events.

The evaluator performs one-to-one event matching within a configurable tolerance window.
It is intentionally independent of the detector and dataset so injected events and
hand-labelled incidents can be evaluated reproducibly.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import pandas as pd


DateLike = str | date | datetime | pd.Timestamp


def _normalise_dates(values: Iterable[DateLike], label: str) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for value in values:
        try:
            stamp = pd.Timestamp(value).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid date: {value!r}") from exc
        if pd.isna(stamp):
            raise ValueError(f"{label} contains an invalid date: {value!r}")
        dates.append(stamp)
    return sorted(set(dates))


def evaluate_events(
    actual_events: Iterable[DateLike],
    detected_events: Iterable[DateLike],
    tolerance_days: int = 0,
) -> dict[str, Any]:
    """Return precision, recall, F1 and detection delay for timestamped events.

    Each actual event can match at most one detection. A detection is eligible when it
    falls between the event date and event date + tolerance_days (inclusive); early
    detections are not credited. When several detections qualify, the earliest is used.
    Duplicate dates are collapsed because the unit of evaluation is an event date.
    """
    if isinstance(tolerance_days, bool) or not isinstance(tolerance_days, int) or tolerance_days < 0:
        raise ValueError("tolerance_days must be a non-negative integer.")

    actual = _normalise_dates(actual_events, "actual_events")
    detected = _normalise_dates(detected_events, "detected_events")
    remaining = set(detected)
    matches: list[dict[str, Any]] = []

    for event_date in actual:
        eligible = sorted(
            prediction
            for prediction in remaining
            if pd.Timedelta(0) <= prediction - event_date <= pd.Timedelta(days=tolerance_days)
        )
        if eligible:
            prediction = eligible[0]
            remaining.remove(prediction)
            matches.append(
                {
                    "actual_date": event_date.date().isoformat(),
                    "detected_date": prediction.date().isoformat(),
                    "delay_days": int((prediction - event_date).days),
                }
            )

    true_positives = len(matches)
    false_positives = len(detected) - true_positives
    false_negatives = len(actual) - true_positives
    precision = true_positives / len(detected) if detected else 0.0
    recall = true_positives / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [match["delay_days"] for match in matches]

    return {
        "actual_events": len(actual),
        "detected_events": len(detected),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_detection_delay_days": sum(delays) / len(delays) if delays else None,
        "matches": matches,
    }
