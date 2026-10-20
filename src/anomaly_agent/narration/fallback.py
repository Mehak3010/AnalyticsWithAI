"""Deterministic, evidence-based summary used when LLM narration fails."""

from __future__ import annotations

from typing import Any

from .facts import to_llm_facts


def template_summary(facts: dict[str, Any]) -> str:
    """Build a four-line summary using only verified facts."""
    f = to_llm_facts(facts)

    metric = str(f.get("metric", "KPI"))
    scope = str(f.get("scope", "overall"))
    date = str(f.get("date", "the latest period"))
    direction = str(f.get("direction") or "changed").lower()
    assessment = str(f.get("assessment", "unknown")).lower()

    where = "overall" if scope == "overall" else f"in {scope}"

    line1 = f"What changed: {metric} {where} {direction} on {date}."

    if "observed" in f and "expected" in f:
        line1 += f" Observed {f['observed']} against an expected {f['expected']}"

        change_percent = f.get("change_percent")
        if change_percent is not None:
            line1 += f" ({change_percent} versus expected)"

        line1 += "."

    drivers = f.get("drivers") or []
    driver_note = str(f.get("driver_note") or "").strip()

    if drivers:
        driver_names = "; ".join(str(driver) for driver in drivers)
        line2 = (
            "Possible driver: "
            f"{driver_names} may have contributed; "
            "these are candidate explanations, not confirmed causes."
        )
    else:
        line2 = "Possible driver: not determined from the data."

    if driver_note:
        line2 += f" {driver_note}"

    assessment_text = {
        "favourable": (
            "The movement aligns with the KPI's configured preferred "
            "direction, but it is still outside the expected range."
        ),
        "unfavourable": (
            "The movement is opposite to the KPI's configured preferred direction and is outside the expected range."
        ),
        "neutral": (
            "The movement is outside the expected range; its business "
            "impact is not classified as favourable or unfavourable."
        ),
        "unknown": ("The movement is outside the expected range, but its business impact has not been classified."),
    }

    line3 = f"Why it matters: {assessment_text.get(assessment, assessment_text['unknown'])}"

    if drivers:
        line4 = "Next check: validate the leading segments and investigate recent changes before assigning a cause."
    else:
        line4 = (
            f"Next check: validate the latest {metric} data for {scope} and compare it with the historical baseline."
        )

    return f"{line1}\n{line2}\n{line3}\n{line4}"
