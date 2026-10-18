"""Turn a raw anomaly dict into a display-ready fact bundle for the LLM and the template.

Numbers are pre-formatted for reading, and statistical fields (z-score, MAD, raw
deviations) are left out, so the model has nothing technical to echo. The same slim
bundle is what the numeric validator checks against.
"""

from __future__ import annotations

from typing import Any

from .prompts import to_jsonable


def fmt_value(value: Any, kind: str | None = None) -> str:
    v = float(value)
    if kind == "ratio":
        return f"{v:.1%}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def fmt_change(value: Any, kind: str | None = None) -> str:
    v = abs(float(value))
    if kind == "ratio":
        return f"{v * 100:.1f} percentage points"
    return f"{v:,.0f}" if v >= 100 else f"{v:,.2f}"


def to_llm_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Slim, readable bundle. This is the only thing the LLM sees."""
    f = to_jsonable(facts)
    kind = f.get("metric_type")  # "ratio" | "additive" | None
    segment = f.get("segment")
    out: dict[str, Any] = {
        "metric": str(f.get("metric", "metric")).replace("_", " "),
        "scope": "overall" if segment in (None, "total") else str(segment),
        "date": f.get("date"),
        "direction": f.get("direction"),
    }
    if f.get("observed") is not None and f.get("expected") is not None:
        out["observed"] = fmt_value(f["observed"], kind)
        out["expected"] = fmt_value(f["expected"], kind)
        out["change"] = fmt_change(float(f["observed"]) - float(f["expected"]), kind)
    if f.get("rel_deviation") is not None:
        out["change_percent"] = f"{abs(float(f['rel_deviation'])):.1%}"
    good = f.get("is_good")
    out["assessment"] = "favourable" if good is True else "unfavourable" if good is False else "neutral"
    out["drivers"] = list(f.get("drivers") or [])
    if f.get("driver_note"):
        out["driver_note"] = str(f["driver_note"])
    return out
