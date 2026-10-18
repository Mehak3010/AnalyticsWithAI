"""Two naive detectors to compare against: a fixed percentage threshold and a plain moving-average z-score.

Neither uses seasonality, robust statistics or persistence, which is the point of the comparison.
Each returns {date: 'up' | 'down'} for the days it flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Flags = dict[pd.Timestamp, str]


def _flags(score: pd.Series, threshold: float) -> Flags:
    hit = score[score.abs() >= threshold].dropna()
    return {pd.Timestamp(d): ("up" if v > 0 else "down") for d, v in hit.items()}


def fixed_threshold_flags(series: pd.Series, pct: float = 0.20, window: int = 28) -> Flags:
    """Flag a day that is at least ``pct`` away from the mean of the previous ``window`` days."""
    ref = series.shift(1).rolling(window, min_periods=window // 2).mean()
    return _flags((series - ref) / ref, pct)


def moving_average_z_flags(series: pd.Series, z: float = 3.5, window: int = 28) -> Flags:
    """Flag a day more than ``z`` standard deviations from the previous ``window`` days' mean."""
    past = series.shift(1).rolling(window, min_periods=window // 2)
    score = (series - past.mean()) / past.std().replace(0, np.nan)
    return _flags(score, z)


def replay_flags(flags: Flags, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, list[tuple[pd.Timestamp, str]]]:
    """Shape per-day flags like a day-by-day replay: on each day the method only reports that day."""
    return {d: ([(d, flags[d])] if d in flags else []) for d in dates}
