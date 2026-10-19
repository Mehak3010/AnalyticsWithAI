"""Segment-level root-cause attribution for headline (total-series) anomalies.

Additive metrics: segment move = actual - baseline, share = move / sum of moves, so shares
add up to 100% within a dimension. Ratio metrics: the change in the overall ratio is split
into a mix effect (volume shifted between segments) and a rate effect (the ratio changed
within segments); the two add up exactly to the total change.

Baseline for a segment = median of the same segment over the previous N equally long
periods, shifted by whole weeks (so the weekday pattern lines up).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

N_BASELINE = 4
COVERAGE = 0.8
MAX_CONTRIBUTORS = 3
MIN_BASELINE_WINDOWS = 2


def decompose_ratio(num1: pd.Series, den1: pd.Series, num0: pd.Series, den0: pd.Series) -> dict[str, Any] | None:
    """Split the change of a ratio (sum num / sum den) into mix and rate effects per segment.

    1 = current period, 0 = baseline; all four series share one segment index.
    mix_i  = (w1 - w0) * (r0 - R0)      (centred on the overall baseline ratio R0)
    rate_i = w1 * (r1 - r0)
    where w is the segment's share of the denominator and r its ratio. Sum of mix + rate
    over all segments equals R1 - R0 exactly. Segments with a zero denominator get a
    neutral filler ratio; they carry zero weight, so the identity still holds.
    """
    total_den1, total_den0 = float(den1.sum()), float(den0.sum())
    if total_den1 <= 0 or total_den0 <= 0:
        return None
    r1_total = float(num1.sum()) / total_den1
    r0_total = float(num0.sum()) / total_den0
    w1, w0 = den1 / total_den1, den0 / total_den0
    r0 = (num0 / den0.where(den0 > 0)).fillna(r0_total)
    r1 = (num1 / den1.where(den1 > 0)).fillna(r0)
    mix = (w1 - w0) * (r0 - r0_total)
    rate = w1 * (r1 - r0)
    table = pd.DataFrame({"mix": mix, "rate": rate, "effect": mix + rate})
    return {"table": table, "R0": r0_total, "R1": r1_total, "total_change": r1_total - r0_total}


class AttributionEngine:
    def __init__(
        self,
        config: Config,
        n_baseline: int = N_BASELINE,
        coverage: float = COVERAGE,
        max_contributors: int = MAX_CONTRIBUTORS,
    ):
        self.config = config
        self.n_baseline = n_baseline
        self.coverage = coverage
        self.max_contributors = max_contributors

    # ------------------------------------------------------------------ public

    def attribute(self, anomaly: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        """Explain a headline anomaly. Returns {} when attribution is not possible."""
        if anomaly.get("segment") != "total":
            return {}
        metric_cfg = next((m for m in self.config.metrics if m.name == anomaly["metric"]), None)
        if metric_cfg is None:
            return {}

        date_col = self.config.dataset.date_column
        windows = self._windows(pd.Timestamp(anomaly["date"]).normalize())
        frame = df[(df[date_col] >= windows[-1][0]) & (df[date_col] < windows[0][1])]
        baseline_windows = sum(1 for lo, hi in windows[1:] if ((frame[date_col] >= lo) & (frame[date_col] < hi)).any())
        if frame.empty or baseline_windows < MIN_BASELINE_WINDOWS:
            return {}

        by_dimension: dict[str, dict[str, Any]] = {}
        for dim in self.config.dataset.dimensions:
            if dim not in frame.columns:
                continue
            try:
                if metric_cfg.type == "ratio":
                    result = self._ratio(frame, dim, windows, metric_cfg)
                else:
                    result = self._additive(frame, dim, windows, metric_cfg)
            except ValueError as e:
                logger.warning("Attribution skipped for %s by %s: %s", anomaly["metric"], dim, e)
                continue
            if result and result["contributors"]:
                by_dimension[dim] = result
        if not by_dimension:
            return {}

        primary = self._pick_primary(by_dimension)
        best = by_dimension[primary]
        is_ratio = metric_cfg.type == "ratio"
        return {
            "metric_type": "ratio" if is_ratio else "additive",
            "primary_dimension": primary,
            "contributors": best["contributors"],
            "coverage": best["coverage"],
            "by_dimension": by_dimension,
            "drivers": [self._driver_text(c, is_ratio) for c in best["contributors"]],
            "note": best.get("note", ""),
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _pick_primary(by_dimension: dict[str, dict[str, Any]]) -> str:
        """The headline dimension is the one whose top contributor moved the most in absolute terms.

        A share-based rule is fooled by noise: in a dimension where the other segments happen to move
        the opposite way, a small segment can look like it explains 100% of the aligned movement.
        The absolute move is not (in simulation: 85% vs 65% correct at a 20% total move).
        """
        return max(by_dimension, key=lambda d: abs(by_dimension[d]["contributors"][0]["move"]))

    def _windows(self, date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """[current, baseline 1, ... baseline N] as (start, end-exclusive); weekly shifts."""
        length = 1 if self.config.dataset.grain == "daily" else 7
        out = []
        for j in range(self.n_baseline + 1):
            end = date - pd.Timedelta(days=7 * j)
            out.append((end - pd.Timedelta(days=length - 1), end + pd.Timedelta(days=1)))
        return out

    def _components(
        self,
        frame: pd.DataFrame,
        dim: str,
        windows: list[tuple[pd.Timestamp, pd.Timestamp]],
        aggregation: str,
        column: str | None,
    ) -> tuple[pd.Series, pd.Series]:
        """Per-segment (current, baseline-median) totals of one component."""
        date_col = self.config.dataset.date_column
        cols = []
        for lo, hi in windows:
            window = frame[(frame[date_col] >= lo) & (frame[date_col] < hi)]
            grouped = window.groupby(dim, observed=True)
            if aggregation == "sum" and column:
                s = grouped[column].sum()
            elif aggregation == "count":
                s = grouped[column].count() if column else grouped.size()
            else:
                raise ValueError(f"unsupported aggregation '{aggregation}'")
            cols.append(s.astype(float))
        table = pd.concat(cols, axis=1).fillna(0.0)
        table.columns = range(len(windows))
        return table[0], table.iloc[:, 1:].median(axis=1)

    def _select(self, effects: pd.Series, total: float, dim: str, extra: pd.DataFrame | None = None):
        """Largest same-direction contributors until `coverage` of the move is explained."""
        sign = 1.0 if total > 0 else -1.0
        aligned = (effects * sign).sort_values(ascending=False)
        aligned = aligned[aligned > 0]
        gross = float(aligned.sum())
        picked: list[dict[str, Any]] = []
        covered = 0.0
        for seg, val in aligned.items():
            # share: portion of the same-direction movement (never above 100%).
            # net_share: effect / net total move (can exceed 100% when other segments offset).
            entry: dict[str, Any] = {
                "segment": f"{dim}={seg}",
                "move": round(float(effects[seg]), 4),
                "share": round(float(val) / gross, 4),
                "net_share": round(float(effects[seg]) / total, 4),
            }
            if extra is not None:
                entry["mix"] = round(float(extra.loc[seg, "mix"]), 6)
                entry["rate"] = round(float(extra.loc[seg, "rate"]), 6)
            picked.append(entry)
            covered += float(val) / gross
            if covered >= self.coverage or len(picked) >= self.max_contributors:
                break
        return picked, round(covered, 4)

    def _additive(self, frame, dim, windows, metric_cfg) -> dict[str, Any] | None:
        cur, base = self._components(frame, dim, windows, metric_cfg.aggregation, getattr(metric_cfg, "column", None))
        move = cur - base
        total = float(move.sum())
        if abs(total) < 1e-12:
            return None
        contributors, covered = self._select(move, total, dim)
        return {"dimension": dim, "total_move": round(total, 4), "contributors": contributors, "coverage": covered}

    def _ratio(self, frame, dim, windows, metric_cfg) -> dict[str, Any] | None:
        num, den = metric_cfg.numerator, metric_cfg.denominator
        num1, num0 = self._components(frame, dim, windows, num.aggregation, getattr(num, "column", None))
        den1, den0 = self._components(frame, dim, windows, den.aggregation, getattr(den, "column", None))
        index = num1.index.union(den1.index)
        num1, num0, den1, den0 = (s.reindex(index, fill_value=0.0) for s in (num1, num0, den1, den0))

        result = decompose_ratio(num1, den1, num0, den0)
        if result is None or abs(result["total_change"]) < 1e-12:
            return None
        table, total = result["table"], result["total_change"]
        contributors, covered = self._select(table["effect"], total, dim, extra=table)

        mix_total, rate_total = float(table["mix"].sum()), float(table["rate"].sum())
        num_pct = float(num1.sum() / num0.sum() - 1) if num0.sum() > 0 else None
        den_pct = float(den1.sum() / den0.sum() - 1) if den0.sum() > 0 else None
        return {
            "dimension": dim,
            "total_change": round(total, 6),
            "mix_effect": round(mix_total, 6),
            "rate_effect": round(rate_total, 6),
            "numerator_change": None if num_pct is None else round(num_pct, 4),
            "denominator_change": None if den_pct is None else round(den_pct, 4),
            "contributors": contributors,
            "coverage": covered,
            "note": self._ratio_note(mix_total, rate_total, num_pct, den_pct),
        }

    @staticmethod
    def _ratio_note(mix: float, rate: float, num_pct: float | None, den_pct: float | None) -> str:
        total = abs(mix) + abs(rate)
        if total == 0:
            return ""
        rate_share = abs(rate) / total
        if rate_share >= 0.5:
            note = (
                f"Mostly a rate effect ({rate_share:.0%}): the ratio changed within segments "
                "rather than volume shifting between them."
            )
        else:
            note = (
                f"Mostly a mix effect ({1 - rate_share:.0%}): volume shifted between segments "
                "rather than the ratio changing within them."
            )
        if num_pct is not None and den_pct is not None:
            note += f" The numerator changed by {num_pct:+.1%} and the denominator by {den_pct:+.1%}."
        return note

    @staticmethod
    def _driver_text(contributor: dict[str, Any], is_ratio: bool) -> str:
        share = f"{abs(contributor['share']):.0%}"
        if not is_ratio:
            return f"{contributor['segment']} ({share} of the move)"
        kind = "rate change" if abs(contributor["rate"]) >= abs(contributor["mix"]) else "mix shift"
        return f"{contributor['segment']} ({share} of the change, mainly a {kind})"
