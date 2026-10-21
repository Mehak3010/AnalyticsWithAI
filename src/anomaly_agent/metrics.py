from typing import Any

import numpy as np
import pandas as pd

from .config import Config


class MetricEngine:
    def __init__(self, config: Config):
        self.config = config
        # First and last dates in the data; set in process() and used to drop partial weeks.
        self._first: pd.Timestamp | None = None
        self._last: pd.Timestamp | None = None

    def process(self, df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
        results = {}
        date_col = self.config.dataset.date_column

        if df.empty:
            return results

        self._first = df[date_col].min().normalize()
        self._last = df[date_col].max().normalize()

        # 1. Total series (headline series)
        for metric_cfg in self.config.metrics:
            series = self._compute_series(df, metric_cfg, date_col, self.config.dataset.grain)
            results[(metric_cfg.name, "total")] = series

        # 2. Segment series. Group once per dimension instead of repeatedly
        # scanning the entire DataFrame for each distinct segment value.
        for dim in self.config.dataset.dimensions:
            for val, segment_df in df.dropna(subset=[dim]).groupby(dim, sort=False):
                segment_name = f"{dim}={val}"

                for metric_cfg in self.config.metrics:
                    series = self._compute_series(
                        segment_df,
                        metric_cfg,
                        date_col,
                        self.config.dataset.segment_grain,
                    )
                    if not series.empty:
                        results[(metric_cfg.name, segment_name)] = series

        return results

    @staticmethod
    def _aggregate(
        df: pd.DataFrame,
        date_col: str,
        freq: str,
        aggregation: str,
        column: str | None = None,
    ) -> pd.Series:
        """Aggregate one column (or count rows) per time bucket."""
        grouped = df.groupby(pd.Grouper(key=date_col, freq=freq))

        if aggregation == "count" and not column:
            return grouped.size()  # count rows, not non-null values of some column
        if not column:
            raise ValueError(f"Aggregation '{aggregation}' needs a column")
        if aggregation == "sum":
            return grouped[column].sum()
        if aggregation == "count":
            return grouped[column].count()
        if aggregation == "nunique":
            return grouped[column].nunique()
        raise ValueError(f"Unknown aggregation: {aggregation}")

    def _trim_partial_weeks(self, series: pd.Series) -> pd.Series:
        """Drop weeks that start before the first or end after the last date in the data.

        Weekly labels are week-ending Sundays, so the week started 6 days earlier.
        Partial weeks have fewer orders and would look like false drops.
        """
        if self._first is None or self._last is None:
            return series
        week_start = series.index - pd.Timedelta(days=6)
        return series[(week_start >= self._first) & (series.index <= self._last)]

    def _compute_series(self, df: pd.DataFrame, metric_cfg: Any, date_col: str, grain: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        freq = "D" if grain == "daily" else "W"

        if metric_cfg.type == "additive":
            series = self._aggregate(df, date_col, freq, metric_cfg.aggregation, metric_cfg.column)

        elif metric_cfg.type == "ratio":
            num_cfg = metric_cfg.numerator
            den_cfg = metric_cfg.denominator
            num = self._aggregate(df, date_col, freq, num_cfg.aggregation, num_cfg.column)
            den = self._aggregate(df, date_col, freq, den_cfg.aggregation, den_cfg.column)
            # Recompute from components (never average ratios); guard against division by zero.
            series = (num / den).replace([np.inf, -np.inf], np.nan)

        else:
            raise ValueError(f"Unknown metric type: {metric_cfg.type}")

        if freq == "W":
            series = self._trim_partial_weeks(series)
        if series.empty:
            return series

        # Reindex to the full calendar so missing dates show up as NaN
        full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
        return series.reindex(full_range)
