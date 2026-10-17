import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .config import Config

class MetricEngine:
    def __init__(self, config: Config):
        self.config = config

    def process(self, df: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
        results = {}
        date_col = self.config.dataset.date_column
        
        # 1. Total series (Headline series)
        for metric_cfg in self.config.metrics:
            series = self._compute_series(df, metric_cfg, date_col, self.config.dataset.grain)
            results[(metric_cfg.name, "total")] = series
            
        # 2. Segment series
        # Limit combinatorial explosion by only looking at one dimension at a time for now
        for dim in self.config.dataset.dimensions:
            unique_values = df[dim].dropna().unique()
            for val in unique_values:
                segment_df = df[df[dim] == val]
                segment_name = f"{dim}={val}"
                
                for metric_cfg in self.config.metrics:
                    series = self._compute_series(
                        segment_df, 
                        metric_cfg, 
                        date_col, 
                        self.config.dataset.segment_grain
                    )
                    if not series.empty:
                        results[(metric_cfg.name, segment_name)] = series
                        
        return results

    def _compute_series(self, df: pd.DataFrame, metric_cfg: Any, date_col: str, grain: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
            
        # Group by grain
        freq = 'D' if grain == 'daily' else 'W'
        
        if metric_cfg.type == "additive":
            if metric_cfg.aggregation == "sum":
                series = df.groupby(pd.Grouper(key=date_col, freq=freq))[metric_cfg.column].sum()
            elif metric_cfg.aggregation == "count":
                series = df.groupby(pd.Grouper(key=date_col, freq=freq))[metric_cfg.column].count()
            else:
                raise ValueError(f"Unknown aggregation: {metric_cfg.aggregation}")
        
        elif metric_cfg.type == "ratio":
            # Numerator
            num_cfg = metric_cfg.numerator
            if num_cfg.aggregation == "sum":
                num = df.groupby(pd.Grouper(key=date_col, freq=freq))[num_cfg.column].sum()
            elif num_cfg.aggregation == "count":
                num = df.groupby(pd.Grouper(key=date_col, freq=freq))[num_cfg.column].count()
            
            # Denominator
            den_cfg = metric_cfg.denominator
            if den_cfg.aggregation == "sum":
                den = df.groupby(pd.Grouper(key=date_col, freq=freq))[den_cfg.column].sum()
            elif den_cfg.aggregation == "count":
                # If column not specified, count rows
                col = den_cfg.column if den_cfg.column else df.columns[0]
                den = df.groupby(pd.Grouper(key=date_col, freq=freq))[col].count()
            
            # Compute ratio
            series = num / den
            # Handle division by zero
            series = series.replace([np.inf, -np.inf], np.nan)
            
        else:
            raise ValueError(f"Unknown metric type: {metric_cfg.type}")
            
        # Reindex to full calendar to handle missing dates
        full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
        series = series.reindex(full_range)
        
        return series
