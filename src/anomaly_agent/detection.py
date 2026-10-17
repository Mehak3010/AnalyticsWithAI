import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from typing import Dict, List, Any, Tuple, Optional
from .config import Config

class Detector:
    def __init__(self, config: Config):
        self.config = config

    def detect(self, series_dict: Dict[Tuple[str, str], pd.Series]) -> List[Dict[str, Any]]:
        anomalies = []
        for (metric, segment), series in series_dict.items():
            if len(series.dropna()) < self.config.detection.min_history:
                continue
            
            # 1. Baseline & Residual
            baseline, residual = self._get_baseline_and_residual(series)
            
            # 2. Score (Robust Z-Score)
            z_scores = self._compute_z_scores(residual)
            
            # 3. Flagging
            expected = baseline
            observed = series
            
            # Flag points
            flags = (np.abs(z_scores) >= self.config.detection.z_threshold)
            
            # Min relative change
            rel_change = np.abs((observed - expected) / expected)
            flags = flags & (rel_change >= self.config.detection.min_relative_change)
            
            # 4. Persistence Rule
            persistence_flags = self._apply_persistence(flags)
            
            # 5. Collect Anomalies
            detected_indices = persistence_flags[persistence_flags].index
            for idx in detected_indices:
                z = z_scores.loc[idx]
                obs = observed.loc[idx]
                exp = expected.loc[idx]
                
                # Determine if direction is good or bad
                direction = "up" if obs > exp else "down"
                metric_cfg = next(m for m in self.config.metrics if m.name == metric)
                
                is_good = False
                if metric_cfg.good_direction == "up" and direction == "up":
                    is_good = True
                elif metric_cfg.good_direction == "down" and direction == "down":
                    is_good = True
                elif metric_cfg.good_direction == "neutral":
                    is_good = True

                anomaly = {
                    "metric": metric,
                    "segment": segment,
                    "date": idx,
                    "observed": float(obs),
                    "expected": float(exp),
                    "abs_deviation": float(np.abs(obs - exp)),
                    "rel_deviation": float(rel_change.loc[idx]),
                    "z_score": float(z),
                    "direction": direction,
                    "is_good": is_good,
                    "priority": metric_cfg.priority
                }
                anomalies.append(anomaly)
                
        return anomalies

    def _get_baseline_and_residual(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        series = series.ffill().bfill()
        
        seasonal_period = self.config.detection.seasonal_period
        # Check if we have at least two full seasonal cycles
        if len(series.dropna()) >= 2 * seasonal_period:
            try:
                stl = STL(series, period=seasonal_period, robust=True)
                res = stl.fit()
                baseline = res.trend + res.seasonal
                residual = res.resid
                return baseline, residual
            except Exception:
                pass
        
        # Fallback to rolling median with day-of-week adjustment
        rolling_median = series.rolling(window=28, min_periods=7).median()
        # Day-of-week adjustment: subtract median of that day of week in the window
        # For simplicity, just use rolling median for now
        baseline = rolling_median
        residual = series - baseline
        return baseline, residual

    def _compute_z_scores(self, residual: pd.Series) -> pd.Series:
        median = residual.median()
        mad = np.median(np.abs(residual - median))
        if mad == 0:
            mad = 1e-6 # Guard against MAD of zero
        
        z = 0.6745 * (residual - median) / mad
        return z

    def _apply_persistence(self, flags: pd.Series) -> pd.Series:
        required = self.config.detection.persistence["required"]
        window = self.config.detection.persistence["window"]
        
        # N of last M points
        persistence = flags.rolling(window=window).sum() >= required
        return persistence
