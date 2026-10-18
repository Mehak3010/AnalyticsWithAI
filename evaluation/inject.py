"""Inject known anomalies, either into an aggregated series or into the raw rows of one segment."""

from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_TYPES = ("spike", "drop", "level_shift", "gradual_drift")


def injected_direction(anomaly_type: str, magnitude: float) -> str:
    """Direction the injected change pushes the series: 'up' or 'down'."""
    return "down" if (anomaly_type == "drop") != (magnitude < 0) else "up"


class AnomalyInjector:
    def __init__(self, seed: int = 42, ramp_days: int = 14):
        self.rng = np.random.default_rng(seed)
        self.ramp_days = ramp_days

    def inject(self, series: pd.Series, anomaly_type: str, date: pd.Timestamp, magnitude: float) -> pd.Series:
        """Return a copy of ``series`` with a known anomaly starting on ``date``.

        spike / drop: one day, multiplied by 1 +/- magnitude.
        level_shift: every day from ``date`` on is multiplied by 1 + magnitude.
        gradual_drift: ramps linearly to 1 + magnitude over ``ramp_days`` days, then holds.
        """
        out = series.astype(float).copy()
        date = pd.Timestamp(date)
        if date not in out.index:
            return out
        idx = out.index.get_loc(date)

        if anomaly_type == "spike":
            out.iloc[idx] *= 1 + magnitude
        elif anomaly_type == "drop":
            out.iloc[idx] *= 1 - magnitude
        elif anomaly_type == "level_shift":
            out.iloc[idx:] *= 1 + magnitude
        elif anomaly_type == "gradual_drift":
            steps = min(self.ramp_days, len(out) - idx)
            ramp = 1 + magnitude * np.arange(1, steps + 1) / self.ramp_days
            out.iloc[idx : idx + steps] *= ramp
            out.iloc[idx + steps :] *= 1 + magnitude
        else:
            raise ValueError(f"Unknown anomaly type '{anomaly_type}'; expected one of {SERIES_TYPES}")
        return out


def inject_segment(
    df: pd.DataFrame,
    date_col: str,
    column: str,
    dim: str,
    value: object,
    date: pd.Timestamp,
    magnitude: float,
) -> tuple[pd.DataFrame, int]:
    """Scale ``column`` by 1 + magnitude for one segment on one day. Returns (new frame, rows changed)."""
    out = df.copy()
    day = pd.Timestamp(date).normalize()
    mask = (out[date_col].dt.normalize() == day) & (out[dim] == value)
    out.loc[mask, column] = out.loc[mask, column] * (1 + magnitude)
    return out, int(mask.sum())
