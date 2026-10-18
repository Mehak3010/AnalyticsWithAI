"""Numeric guardrail: every number in LLM text must exist in the fact bundle.

The LLM only turns verified facts into prose. This module checks that it did not
invent or distort any figure. Matching is tolerant of rounding and of the usual
re-expressions (a share of 0.7 written as "70%", a drop of -9.4 written as "9%").
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.(?P<dec>\d+))?"
    r"\s?(?P<suffix>%|k|K|M|bn|million|billion|thousand)?"
    r"(?![A-Za-z])"
)

_MULTIPLIERS = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "billion": 1e9,
}


@dataclass(frozen=True)
class ParsedNumber:
    """A number found in text, with the precision it was written at."""

    raw: str
    value: float
    decimals: int


@dataclass
class ValidationResult:
    ok: bool
    unmatched: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[ParsedNumber]:
    """Return every number in ``text`` (sign ignored; k/M/bn suffixes expanded)."""
    found: list[ParsedNumber] = []
    for m in _NUMBER_RE.finditer(text):
        whole = m.group("int").replace(",", "")
        dec = m.group("dec") or ""
        value = float(f"{whole}.{dec}" if dec else whole)
        suffix = m.group("suffix")
        if suffix and suffix != "%":
            value *= _MULTIPLIERS[suffix.lower() if suffix != "M" else "m"]
        found.append(ParsedNumber(raw=m.group(0).strip(), value=value, decimals=len(dec)))
    return found


def _walk(obj: Any) -> Iterable[float]:
    """Yield every numeric value in a nested structure, including numbers inside strings."""
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, str):
        for n in extract_numbers(obj):
            yield n.value
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(obj, list | tuple | set):
        for v in obj:
            yield from _walk(v)
    else:
        # numpy scalars, Timestamps, dates, etc.
        try:
            yield float(obj)
        except (TypeError, ValueError):
            yield from _walk(str(obj))


def allowed_values(fact_bundle: Any, extra: Iterable[float] = ()) -> set[float]:
    """Absolute values a text number may legitimately match.

    Each bundle number v also allows |v| and |v| * 100 (fraction -> percent).
    """
    allowed: set[float] = set()
    for v in list(_walk(fact_bundle)) + list(extra):
        a = abs(v)
        allowed.add(a)
        allowed.add(a * 100)
    return allowed


def _matches(num: ParsedNumber, allowed: set[float], rel_tol: float) -> bool:
    abs_tol = 0.5 * 10 ** (-num.decimals)  # rounding to the written precision
    return any(abs(num.value - c) <= max(abs_tol, rel_tol * abs(c)) for c in allowed)


def validate_numbers(
    text: str,
    fact_bundle: Any,
    rel_tol: float = 0.01,
    extra_allowed: Iterable[float] = (),
) -> ValidationResult:
    """Check every number in ``text`` against ``fact_bundle``.

    ``rel_tol`` is the relative tolerance; the absolute tolerance is half a unit
    of the last written digit (so "9%" matches 9.4 and "9.4%" does not match 9.9).
    """
    allowed = allowed_values(fact_bundle, extra_allowed)
    unmatched = [n.raw for n in extract_numbers(text) if not _matches(n, allowed, rel_tol)]
    return ValidationResult(ok=not unmatched, unmatched=unmatched)
