"""Volume-price-mix variance bridge for segmented business metrics.

Inputs are mappings keyed by segment. Each segment value must contain volume and price.
The bridge reconciles to actual revenue minus baseline revenue within floating-point tolerance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .config import Config


class VarianceEngine:
    """Calculate actual-vs-baseline variance and an optional volume-price-mix bridge."""

    def __init__(self, config: Config):
        self.config = config

    def calculate_bridge(
        self,
        actual: Mapping[str, Mapping[str, float]],
        forecast: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        """Decompose revenue variance across segments.

        Each mapping uses segment names as keys and contains numeric volume and price.
        Forecast is the baseline; actual is the current period.
        """
        actual_df = self._normalise(actual, "actual")
        forecast_df = self._normalise(forecast, "forecast")

        segments = sorted(set(actual_df.index) | set(forecast_df.index), key=str)
        actual_df = actual_df.reindex(segments, fill_value=0.0)
        forecast_df = forecast_df.reindex(segments, fill_value=0.0)

        q1, p1 = actual_df["volume"].astype(float), actual_df["price"].astype(float)
        q0, p0 = forecast_df["volume"].astype(float), forecast_df["price"].astype(float)

        if (q1 < 0).any() or (q0 < 0).any():
            raise ValueError("Volume must be non-negative.")
        if (p1 < 0).any() or (p0 < 0).any():
            raise ValueError("Price must be non-negative.")

        total_q1, total_q0 = float(q1.sum()), float(q0.sum())
        share0 = q0 / total_q0 if total_q0 > 0 else pd.Series(0.0, index=segments)
        share1 = q1 / total_q1 if total_q1 > 0 else pd.Series(0.0, index=segments)

        baseline_revenue = float((q0 * p0).sum())
        actual_revenue = float((q1 * p1).sum())
        baseline_avg_price = baseline_revenue / total_q0 if total_q0 > 0 else 0.0

        # Volume effect uses baseline average price; mix uses baseline segment prices;
        # price effect applies each segment's price change to actual volume.
        volume_effect = (total_q1 - total_q0) * baseline_avg_price
        mix_by_segment = total_q1 * (share1 - share0) * p0
        mix_effect = float(mix_by_segment.sum())
        price_by_segment = q1 * (p1 - p0)
        price_effect = float(price_by_segment.sum())

        bridge_total = float(volume_effect + mix_effect + price_effect)
        variance = actual_revenue - baseline_revenue
        tolerance = 1e-7 * max(1.0, abs(variance), abs(actual_revenue))
        if abs(bridge_total - variance) > tolerance:
            raise ArithmeticError("Volume-price-mix bridge failed to reconcile to revenue variance.")

        details = [
            {
                "segment": str(segment),
                "forecast_volume": float(q0.loc[segment]),
                "actual_volume": float(q1.loc[segment]),
                "forecast_price": float(p0.loc[segment]),
                "actual_price": float(p1.loc[segment]),
                "volume_share_change": float(share1.loc[segment] - share0.loc[segment]),
                "mix_effect": float(mix_by_segment.loc[segment]),
                "price_effect": float(price_by_segment.loc[segment]),
            }
            for segment in segments
        ]
        return {
            "baseline_revenue": baseline_revenue,
            "actual_revenue": actual_revenue,
            "variance": variance,
            "variance_pct": variance / baseline_revenue if baseline_revenue else None,
            "bridge": {
                "volume": float(volume_effect),
                "mix": mix_effect,
                "price": price_effect,
                "total": bridge_total,
            },
            "segments": details,
        }

    @staticmethod
    def _normalise(values: Mapping[str, Mapping[str, float]], label: str) -> pd.DataFrame:
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} must map segment names to volume/price values.")

        rows: dict[str, dict[str, float]] = {}
        for segment, metrics in values.items():
            if not isinstance(metrics, Mapping):
                raise TypeError(f"{label}[{segment!r}] must be a mapping.")
            if "volume" not in metrics or "price" not in metrics:
                raise ValueError(f"{label}[{segment!r}] must include volume and price.")
            try:
                volume, price = float(metrics["volume"]), float(metrics["price"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}[{segment!r}] volume and price must be numeric.") from exc
            if not math.isfinite(volume) or not math.isfinite(price):
                raise ValueError(f"{label}[{segment!r}] volume and price must be finite.")
            rows[str(segment)] = {"volume": volume, "price": price}

        return pd.DataFrame.from_dict(rows, orient="index", columns=["volume", "price"]).fillna(0.0)
