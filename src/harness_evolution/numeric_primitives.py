"""Small deterministic building blocks for numeric verification.

These helpers deliberately use finite vocabularies and conversion tables.  A
Skill YAML may select them, but cannot supply executable code, arbitrary
regular expressions, formulas, or network lookups.
"""
from __future__ import annotations

import re
from typing import Iterable


# Canonical concept names are intentionally narrow.  Unknown or ambiguous
# wording is not guessed; callers should return ``unknown`` instead.
ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "participants": ("参与者", "患者", "受试者", "participants", "patients"),
    "samples": ("样本", "samples"),
    "users": ("用户", "users"),
    "hospitalization_rate": ("住院率", "hospitalization rate", "hospitalisation rate"),
    "retention_rate": ("留存率", "retention rate"),
    "revenue": ("收入", "营收", "revenue"),
    "success_rate": ("成功率", "success rate"),
    "weight": ("重量", "净重", "weight"),
    "distance": ("距离", "distance"),
    "temperature": ("温度", "temperature"),
}

_CLAUSE_BOUNDARY_RE = re.compile(r"[，,；;。.!！？?\n]")
_QUALIFIER_PATTERNS = (
    re.compile(r"(?<![A-Za-z])([A-Z])\s*(?:产品|组|product|group)", re.I),
    re.compile(r"([甲乙丙丁])\s*组"),
)

# unit -> (dimension, canonical unit, multiplier to canonical unit)
MEASUREMENT_UNITS: dict[str, tuple[str, str, float]] = {
    "千克": ("mass", "kg", 1.0),
    "公斤": ("mass", "kg", 1.0),
    "kg": ("mass", "kg", 1.0),
    "克": ("mass", "kg", 0.001),
    "g": ("mass", "kg", 0.001),
    "吨": ("mass", "kg", 1000.0),
    "公里": ("distance", "km", 1.0),
    "千米": ("distance", "km", 1.0),
    "km": ("distance", "km", 1.0),
    "米": ("distance", "km", 0.001),
    "m": ("distance", "km", 0.001),
    "°c": ("temperature", "C", 1.0),
    "℃": ("temperature", "C", 1.0),
}


def entity_signature(text: str, start: int, end: int) -> tuple[str, ...]:
    """Return conservative entity/metric anchors around one numeric span."""

    left = max((match.end() for match in _CLAUSE_BOUNDARY_RE.finditer(text, 0, start)), default=0)
    right_match = _CLAUSE_BOUNDARY_RE.search(text, end)
    right = right_match.start() if right_match else len(text)
    clause = text[left:right].strip().lower()
    concepts = [
        name
        for name, aliases in ENTITY_ALIASES.items()
        if any(alias.lower() in clause for alias in aliases)
    ]
    qualifiers: list[str] = []
    for pattern in _QUALIFIER_PATTERNS:
        qualifiers.extend(match.group(1).upper() for match in pattern.finditer(clause))
    return tuple(sorted(dict.fromkeys([*concepts, *qualifiers])))


def entities_compatible(left: Iterable[str], right: Iterable[str]) -> bool:
    """Require a shared concept and, when present, the same entity qualifier."""

    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return False
    concepts = set(ENTITY_ALIASES)
    if not (left_set & right_set & concepts):
        return False
    left_qualifiers = left_set - concepts
    right_qualifiers = right_set - concepts
    if left_qualifiers or right_qualifiers:
        return bool(left_qualifiers and right_qualifiers and left_qualifiers & right_qualifiers)
    return True


def concepts_compatible(left: Iterable[str], right: Iterable[str]) -> bool:
    """Whether two facts mention the same allow-listed metric concept."""

    concepts = set(ENTITY_ALIASES)
    return bool(set(left) & set(right) & concepts)


def canonical_measurement(value: float, unit: str) -> tuple[str, str, float] | None:
    """Convert an allow-listed measurement to its canonical dimension/unit."""

    spec = MEASUREMENT_UNITS.get(str(unit).strip().lower())
    if spec is None:
        return None
    dimension, canonical_unit, multiplier = spec
    return dimension, canonical_unit, float(value) * multiplier
