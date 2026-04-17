"""
Requirements natural-language parser.

Parses requirement text into a structured ParsedRequirement.
Handles quantitative bounds (upper/lower/range) and qualitative directives
without any external ML or numerical dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when a requirement text cannot be parsed into a structured form."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedRequirement:
    text: str
    direction: str                    # "upper_bound" | "lower_bound" | "range" | "qualitative"
    value: Optional[float]            # primary (or lower) numeric value; None for qualitative
    value_upper: Optional[float]      # upper value for range direction; None otherwise
    unit: Optional[str]               # unit string; None for qualitative
    condition_clause: Optional[str]   # e.g. "under load", "at rated speed"
    is_qualitative: bool


# ---------------------------------------------------------------------------
# Recognised unit tokens (ordered longest-first to avoid partial matches)
# ---------------------------------------------------------------------------

_UNITS: list[str] = sorted([
    "kN*m", "N*m",
    "m/s^2", "m/s",
    "km/h",
    "degC", "degF",
    "kPa", "MPa", "GPa", "Pa",
    "kHz", "MHz", "Hz",
    "kcal/mol",
    "kN", "MN", "N",
    "kJ", "MJ", "J",
    "kW", "MW", "W",
    "kg", "mg", "g",
    "km", "mm", "cm", "m",
    "ms", "us", "min",
    "dB", "eV",
    "rad", "deg",
    "rpm", "K", "h", "s",
    "cycles", "1",
], key=len, reverse=True)

# Build a single regex alternation that matches any recognised unit token.
# We require a word boundary or end-of-string after the unit so that
# e.g. "mm" doesn't match inside "mmHg".
_UNIT_PATTERN = (
    r"(?P<unit>"
    + "|".join(re.escape(u) for u in _UNITS)
    + r")(?:\b|$)"
)

# ---------------------------------------------------------------------------
# Condition clause keywords  (matched before the main numeric patterns)
# ---------------------------------------------------------------------------

_CONDITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bunder\s+\w+(?:\s+\w+)?", re.IGNORECASE),
    re.compile(r"\bat\s+\w+(?:\s+\w+)?", re.IGNORECASE),
    re.compile(r"\bduring\s+\w+(?:\s+\w+)?", re.IGNORECASE),
    re.compile(r"\bwhen\s+\w+(?:\s+\w+)?", re.IGNORECASE),
    re.compile(r"\bwhile\s+\w+(?:\s+\w+)?", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Qualitative keywords — presence means "qualitative" direction
# ---------------------------------------------------------------------------

_QUALITATIVE_KEYWORDS = frozenset([
    "minimize", "minimise", "maximise", "maximize",
    "minimized", "maximized",
    "optimal", "optimise", "optimize",
    "improve", "reduce", "increase",
    "avoid", "prevent", "ensure",
])

# ---------------------------------------------------------------------------
# Main compiled patterns
# ---------------------------------------------------------------------------

# "between X and Y <unit>"  — range
_RANGE_PATTERN = re.compile(
    r"\bbetween\s+(?P<lo>\d+(?:\.\d+)?)\s+and\s+(?P<hi>\d+(?:\.\d+)?)\s+" + _UNIT_PATTERN,
    re.IGNORECASE,
)

# "< X <unit>"  or "not exceed X <unit>"  or "no more than X <unit>"  — upper_bound
_UPPER_PATTERN = re.compile(
    r"(?:"
    r"(?:shall\s+)?not\s+exceed\s+|"
    r"(?:shall\s+)?(?:be\s+)?(?:no\s+more\s+than|at\s+most|less\s+than\s+or\s+equal\s+to|≤|<=|<|≤)\s*|"
    r"(?:maximum|max)[\s:]+|"
    r"<\s*"
    r")"
    r"(?P<val>\d+(?:\.\d+)?)\s*" + _UNIT_PATTERN,
    re.IGNORECASE,
)

# "> X <unit>" or "at least X <unit>" or "greater than X <unit>"  — lower_bound
_LOWER_PATTERN = re.compile(
    r"(?:"
    r"(?:shall\s+)?(?:be\s+)?(?:at\s+least|greater\s+than\s+or\s+equal\s+to|≥|>=|>|≥)\s*|"
    r"(?:minimum|min)[\s:]+|"
    r">\s*"
    r")"
    r"(?P<val>\d+(?:\.\d+)?)\s*" + _UNIT_PATTERN,
    re.IGNORECASE,
)

# Bare number + unit anywhere in the text (fallback)
_BARE_NUMBER_PATTERN = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*" + _UNIT_PATTERN
)


def _extract_condition(text: str) -> Optional[str]:
    """Return the first condition clause found, or None."""
    for pat in _CONDITION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    return None


def parse(text: str) -> ParsedRequirement:
    """
    Parse a requirement string into a ParsedRequirement.

    Raises ParseError if the text is not quantifiable and contains no
    qualitative keyword.
    """
    stripped = text.strip()
    condition_clause = _extract_condition(stripped)

    # 1. Range pattern: "between X and Y unit"
    m = _RANGE_PATTERN.search(stripped)
    if m:
        return ParsedRequirement(
            text=stripped,
            direction="range",
            value=float(m.group("lo")),
            value_upper=float(m.group("hi")),
            unit=m.group("unit"),
            condition_clause=condition_clause,
            is_qualitative=False,
        )

    # 2. Upper-bound pattern
    m = _UPPER_PATTERN.search(stripped)
    if m:
        return ParsedRequirement(
            text=stripped,
            direction="upper_bound",
            value=float(m.group("val")),
            value_upper=None,
            unit=m.group("unit"),
            condition_clause=condition_clause,
            is_qualitative=False,
        )

    # 3. Lower-bound pattern
    m = _LOWER_PATTERN.search(stripped)
    if m:
        return ParsedRequirement(
            text=stripped,
            direction="lower_bound",
            value=float(m.group("val")),
            value_upper=None,
            unit=m.group("unit"),
            condition_clause=condition_clause,
            is_qualitative=False,
        )

    # 4. Qualitative keywords
    lower = stripped.lower()
    if any(kw in lower for kw in _QUALITATIVE_KEYWORDS):
        return ParsedRequirement(
            text=stripped,
            direction="qualitative",
            value=None,
            value_upper=None,
            unit=None,
            condition_clause=condition_clause,
            is_qualitative=True,
        )

    # 5. Bare number + unit (fallback — treat as upper_bound if "not" present,
    #    otherwise raise ParseError so caller can decide)
    m = _BARE_NUMBER_PATTERN.search(stripped)
    if m:
        # If we found a number+unit but no clear direction keyword, we cannot
        # determine the bound direction — raise ParseError.
        raise ParseError(
            f"Requirement has a quantity but ambiguous direction: {stripped!r}"
        )

    # 6. No number and no qualitative keyword → not parseable
    raise ParseError(
        f"Requirement is not parseable (no quantity and no qualitative keyword): {stripped!r}"
    )
