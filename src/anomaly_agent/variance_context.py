"""Attach a reconciled volume-price-mix bridge to additive sales anomalies."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .variance import VarianceEngine

logger = logging.getLogger(__name__)
BASELINE_WEEKS = 4


def add_variance_context(
    anomalies: list[dict[str, Any]], df: pd.DataFrame, config: Any
) -> None:
    """Attach best-dimension variance context to total sales anomalies, fail-open.

    Effective price is computed as net sales / quantity, not raw list price, so
    the bridge reconciles to the configured net-sales metric despite discounts.
    The baseline uses the median of the same weekday (or same-length week) over
    the previous four weeks.
    """
    sales_metric = next(
        (m for m in config.metrics if m.name == "sales" and m.type == "additive"),
        None,
    )
    date_col = config.dataset.date_column
    if sales_metric is None or not {"sales", "quantity"}.issubset(df.columns):
        return

    engine = VarianceEngine(config)
    length = 1 if config.dataset.grain == "daily" else 7
    for anomaly in anomalies:
        if anomaly.get("segment") != "total" or anomaly.get("metric") != "sales":
            continue
        try:
            end = pd.Timestamp(anomaly["date"]).normalize() + pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=length)
            windows = [
                (start - pd.Timedelta(days=7 * i), end - pd.Timedelta(days=7 * i))
                for i in range(BASELINE_WEEKS + 1)
            ]
            window_data = []
            for lo, hi in windows:
                part = df[(df[date_col] >= lo) & (df[date_col] < hi)]
                window_data.append(part)

            candidates = []
            for dim in config.dataset.dimensions:
                if dim not in df.columns:
                    continue
                def components(
                    part: pd.DataFrame, dimension: str
                ) -> dict[str, dict[str, float]]:
                    grouped = part.groupby(dimension, observed=True).agg(
                        revenue=("sales", "sum"), volume=("quantity", "sum")
                    )
                    result = {}
                    for segment, row in grouped.iterrows():
                        volume = float(row["volume"])
                        revenue = float(row["revenue"])
                        if volume < 0 or not pd.notna(volume) or not pd.notna(revenue):
                            continue
                        price = revenue / volume if volume > 0 else 0.0
                        result[str(segment)] = {"volume": volume, "price": price}
                    return result

                actual = components(window_data[0], dim)
                baseline_parts = [
                    components(part, dim) for part in window_data[1:]
                ]
                segments = set().union(*(part.keys() for part in baseline_parts), actual.keys())
                baseline = {}
                for segment in segments:
                    volumes = [part.get(segment, {}).get("volume", 0.0) for part in baseline_parts]
                    revenues = [
                        part.get(segment, {}).get("volume", 0.0)
                        * part.get(segment, {}).get("price", 0.0)
                        for part in baseline_parts
                    ]
                    volume = float(pd.Series(volumes).median())
                    revenue = float(pd.Series(revenues).median())
                    baseline[segment] = {
                        "volume": volume,
                        "price": revenue / volume if volume > 0 else 0.0,
                    }
                if not actual or not baseline:
                    continue
                bridge = engine.calculate_bridge(actual, baseline)
                candidates.append((abs(bridge["variance"]), dim, bridge))

            if candidates:
                _, dimension, bridge = max(candidates, key=lambda item: item[0])
                anomaly["variance_context"] = {"dimension": dimension, **bridge}
        except Exception as exc:
            logger.warning(
                "Variance context failed for %s: %s: %s",
                anomaly.get("metric"), type(exc).__name__, exc,
            )
