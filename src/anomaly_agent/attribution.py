from typing import Any

import pandas as pd

from .config import Config


class AttributionEngine:
    def __init__(self, config: Config):
        self.config = config

    def attribute(self, anomaly: dict[str, Any], series_dict: dict[tuple[str, str], pd.Series]) -> dict[str, Any]:
        metric = anomaly["metric"]
        anomaly["date"]

        # Only attribute for "total" anomalies
        if anomaly["segment"] != "total":
            return {}

        metric_cfg = next(m for m in self.config.metrics if m.name == metric)

        if metric_cfg.type == "additive":
            return self._attribute_additive(anomaly, series_dict)
        elif metric_cfg.type == "ratio":
            return self._attribute_ratio(anomaly, series_dict)
        else:
            return {}

    def _attribute_additive(
        self, anomaly: dict[str, Any], series_dict: dict[tuple[str, str], pd.Series]
    ) -> dict[str, Any]:
        metric = anomaly["metric"]
        date = anomaly["date"]
        total_move = anomaly["observed"] - anomaly["expected"]

        if total_move == 0:
            return {"top_contributors": []}

        contributions = []
        for (m, seg), series in series_dict.items():
            for (m, seg), series in series_dict.items():
                if m == metric and seg != "total" and date in series.index:
                    # Simplified: use segment's value at this date.
                    # A better way would be to run the same baseline logic for each segment.
                    val = series.loc[date]
                    if pd.isna(val):
                        val = 0

                    # We don't have segment expected here easily,
                    # so we'll just look at the raw values for now.
                    # In a full implementation, we'd use (seg_obs - seg_exp) / total_move.
                    contributions.append(
                        {
                            "segment": seg,
                            "contribution": float(
                                val / anomaly["observed"]
                            ),  # This is share of total, not contribution to move
                        }
                    )

        # Sort and take top
        contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
        return {"top_contributors": contributions[:5]}

    def _attribute_ratio(
        self, anomaly: dict[str, Any], series_dict: dict[tuple[str, str], pd.Series]
    ) -> dict[str, Any]:
        # Ratio attribution is more complex (mix vs rate effect)
        # For now, return a placeholder
        return {"ratio_effect": "mix vs rate analysis pending"}
