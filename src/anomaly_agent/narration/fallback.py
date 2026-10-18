"""Deterministic template summary, used when the LLM is unavailable or fails validation."""

from __future__ import annotations

from typing import Any

from .facts import to_llm_facts


def template_summary(facts: dict[str, Any]) -> str:
    """Four-line summary built only from the slim bundle, so every number is verifiable."""
    f = to_llm_facts(facts)
    where = "overall" if f["scope"] == "overall" else f"in {f['scope']}"
    direction = f.get("direction") or "moved"

    line1 = f"What changed: {f['metric']} {where} moved {direction} on {f.get('date', 'the latest period')}."
    if "observed" in f:
        line1 += f" Observed {f['observed']} against an expected {f['expected']}"
        if "change_percent" in f:
            line1 += f" ({f['change_percent']} {'above' if direction == 'up' else 'below'} expected)"
        line1 += "."

    drivers = f["drivers"]
    if drivers:
        line2 = "Possible driver: " + "; ".join(str(d) for d in drivers) + " may have contributed."
    else:
        line2 = "Possible driver: not determined from the data."
    if f.get("driver_note"):
        line2 += f" {f['driver_note']}"

    verdict = {
        "favourable": "the move is favourable, but outside the normal range.",
        "unfavourable": "the move is unfavourable and outside the normal range.",
    }.get(f["assessment"], "the move is outside the normal range.")
    line3 = f"Why it matters: {verdict}"

    if drivers:
        line4 = "Next check: review the listed segments for recent changes."
    else:
        line4 = f"Next check: review recent data for {f['scope']} {f['metric']}."
    return f"{line1}\n{line2}\n{line3}\n{line4}"
