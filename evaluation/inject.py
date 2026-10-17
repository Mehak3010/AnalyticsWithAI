import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

class AnomalyInjector:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def inject(self, series: pd.Series, anomaly_type: str, date: pd.Timestamp, magnitude: float) -> pd.Series:
        series = series.copy()
        if date not in series.index:
            return series
            
        idx = series.index.get_loc(date)
        
        if anomaly_type == "spike":
            series.iloc[idx] *= (1 + magnitude)
        elif anomaly_type == "drop":
            series.iloc[idx] *= (1 - magnitude)
        elif anomaly_type == "level_shift":
            series.iloc[idx:] *= (1 + magnitude)
        elif anomaly_type == "gradual_drift":
            for i in range(idx, len(series)):
                series.iloc[i] *= (1 + magnitude * (i - idx + 1) / (len(series) - idx))
                
        return series
