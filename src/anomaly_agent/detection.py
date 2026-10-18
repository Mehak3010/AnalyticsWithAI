import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import Config

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, config: Config):
        self.config = config

    def detect(
        self,
        series_dict: dict[tuple[str, str], pd.Series],
        as_of: pd.Timestamp | None = None,
        window: int | None = None,
    ) -> list[dict[str, Any]]:
        """Detect anomalies.

        as_of:  ignore data after this date (no look-ahead when replaying history).
        window: keep only anomalies in the last `window` periods of each series.
                None scans the whole history (use for evaluation, not for alerting).
        """
        cfg = self.config.detection
        anomalies: list[dict[str, Any]] = []
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else None

        for (metric, segment), series in series_dict.items():
            if as_of_ts is not None:
                series = series[series.index <= as_of_ts]
            if len(series.dropna()) < cfg.min_history:
                continue

            metric_cfg = next(m for m in self.config.metrics if m.name == metric)
            period_days = self._period_days(series)

            # Alerting scans (window set) judge the recent window against a baseline fitted
            # WITHOUT it, so a multi-day shift cannot pull its own baseline along with it.
            holdout = window or 0
            baseline, residual, n_ref = self._get_baseline_and_residual(series, holdout)
            z_scores = self._compute_z_scores(residual, baseline, n_ref)
            expected, observed = baseline, series

            # Relative change; undefined (NaN) where the expected value is ~0
            safe = expected.abs() > 1e-9
            rel_change = ((observed - expected).abs() / expected.abs()).where(safe)

            flags = (z_scores.abs() >= cfg.z_threshold) & (rel_change >= cfg.min_relative_change)
            # Persistence must be direction-consistent: N of the last M points flagged in
            # the SAME direction (an up-spike plus an unrelated dip is not persistence).
            is_up = (observed > expected).fillna(False)
            persistent = self._apply_persistence(flags & is_up) | self._apply_persistence(flags & ~is_up)
            # An extreme single point alerts immediately (spike override)
            spike_z = cfg.spike_override_z if cfg.spike_override_z is not None else 3 * cfg.z_threshold
            extreme = (z_scores.abs() >= spike_z) & (rel_change >= cfg.min_relative_change)
            persistent = persistent | extreme

            idxs = list(persistent[persistent].index)
            if window is not None and idxs:
                cutoff = series.index.max() - pd.Timedelta(days=window * period_days)
                idxs = [i for i in idxs if i > cutoff]

            for idx in idxs:
                obs, exp = float(observed.loc[idx]), float(expected.loc[idx])
                direction = "up" if obs > exp else "down"
                is_good = metric_cfg.good_direction == "neutral" or metric_cfg.good_direction == direction
                anomalies.append(
                    {
                        "metric": metric,
                        "segment": segment,
                        "date": idx,
                        "period_days": period_days,
                        "observed": obs,
                        "expected": exp,
                        "abs_deviation": abs(obs - exp),
                        "rel_deviation": float(rel_change.loc[idx]),
                        "z_score": float(z_scores.loc[idx]),
                        "direction": direction,
                        "is_good": is_good,
                        "priority": metric_cfg.priority,
                    }
                )
        return anomalies

    @staticmethod
    def _period_days(series: pd.Series) -> int:
        if len(series.index) < 2:
            return 1
        return max(1, round(pd.Series(series.index).diff().dt.days.median()))

    def _get_baseline_and_residual(self, series: pd.Series, holdout: int = 0) -> tuple[pd.Series, pd.Series, int]:
        """Return (baseline, residual, n_ref).

        n_ref is the number of leading points used as the reference (everything except
        the last `holdout` points). Scale statistics come from the reference only.
        """
        series = series.ffill().bfill()
        n = len(series)

        # Weekly (or coarser) series: the configured seasonal_period is in days and does
        # not apply. Use a trailing median of the previous 8 periods.
        if self._period_days(series) >= 7:
            baseline = series.shift(1).rolling(window=8, min_periods=4).median()
            return baseline, series - baseline, n

        period = self.config.detection.seasonal_period
        n_ref = n - holdout
        if n_ref >= 2 * period + 14:
            try:
                ref = series.iloc[:n_ref]
                # STL estimates the repeating (weekly) pattern; a trailing median of the
                # deseasonalised values gives a robust level that is never fitted on the
                # points being judged. Scale statistics use the same estimator on history.
                seasonal = STL(ref, period=period, robust=True).fit().seasonal
                deseason = ref - seasonal
                level_hist = deseason.shift(1).rolling(window=28, min_periods=14).median()
                fitted = level_hist + seasonal
                if holdout:
                    level = float(deseason.iloc[-28:].median())
                    cycle = seasonal.iloc[-period:].to_numpy()
                    future = pd.Series(
                        [level + cycle[j % period] for j in range(holdout)],
                        index=series.index[n_ref:],
                    )
                    baseline = pd.concat([fitted, future])
                else:
                    baseline = fitted
                return baseline, series - baseline, n_ref
            except Exception as e:
                logger.warning("STL baseline failed, using rolling median: %s", e)

        baseline = series.shift(1).rolling(window=28, min_periods=7).median()
        return baseline, series - baseline, max(n_ref, 1)

    def _compute_z_scores(self, residual: pd.Series, baseline: pd.Series, n_ref: int | None = None) -> pd.Series:
        ref = residual.iloc[:n_ref] if n_ref else residual
        r = ref.dropna()
        median = r.median()
        mad = np.median(np.abs(r - median))
        level = float(np.nanmedian(np.abs(baseline.iloc[:n_ref] if n_ref else baseline)))
        # 1.4826 * MAD is the robust standard deviation; floor it so near-zero MAD
        # cannot create enormous z-scores.
        scale = max(1.4826 * mad, self.config.detection.mad_floor_pct * level, 1e-9)
        return (residual - median) / scale

    def _apply_persistence(self, flags: pd.Series) -> pd.Series:
        required = self.config.detection.persistence["required"]
        window = self.config.detection.persistence["window"]
        # N of the last M points flagged, and the current point itself must be flagged
        # (flags is boolean, for one series and one direction)
        return (flags.astype(int).rolling(window=window).sum() >= required) & flags
