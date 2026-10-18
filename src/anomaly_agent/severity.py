from typing import Any

from .config import Config


class SeverityScorer:
    def __init__(self, config: Config):
        self.config = config

    def score(self, anomaly: dict[str, Any]) -> str:
        z = abs(anomaly["z_score"])
        rel_dev = anomaly["rel_deviation"]
        priority = anomaly["priority"]

        # Simple rule-based severity
        # (This can be more complex as per Section 6.5)

        score = 0

        # Z-score contribution
        if z > 10:
            score += 4
        elif z > 5:
            score += 2
        elif z > 3:
            score += 1

        # Relative deviation contribution
        if rel_dev > 0.5:
            score += 3
        elif rel_dev > 0.2:
            score += 2
        elif rel_dev > 0.1:
            score += 1

        # Priority contribution
        if priority == "high":
            score += 3
        elif priority == "medium":
            score += 1

        # Mapping score to severity
        if score >= 12:
            return "critical"
        elif score >= 8:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"
