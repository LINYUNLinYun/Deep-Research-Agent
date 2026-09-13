"""Declarative, side-effect-free numeric claim verification skill."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.evidence.schemas import Claim, Evidence, VerificationStatus
from src.harness_evolution.numeric_primitives import (
    MEASUREMENT_UNITS,
    canonical_measurement,
    concepts_compatible,
    entities_compatible,
    entity_signature,
)


ALLOWED_PRIMITIVES = {
    "detect_numeric_claim",
    "normalize_number",
    "normalize_percentage",
    "normalize_currency_unit",
    "normalize_date_or_fiscal_year",
    "rank_primary_sources",
    "locate_source_span",
    "compare_value_unit_date",
    "resolve_source_conflict",
    "classify_supported_contradicted_unknown",
    "align_entity_value",
    "convert_measurement_unit",
    "filter_semantically_irrelevant_numbers",
}
COMPOSABLE_PRIMITIVES = {
    "align_entity_value",
    "convert_measurement_unit",
    "filter_semantically_irrelevant_numbers",
}

_NUMERIC_RE = re.compile(
    r"(?P<currency>US\$|HK\$|CNY|RMB|USD|EUR|GBP|JPY|[$€£¥])?\s*"
    r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    # Do not consume the unit prefix in ``千克``/``kg`` as a numeric scale.
    r"(?P<scale>万|亿|千(?!克)|百|trillion|billion|million|thousand|[KMBT](?![A-Za-z]))?\s*"
    r"(?P<percent>%|％|percent|percentage)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:(?:FY|财年)\s*)?(?P<year>19\d{2}|20\d{2})(?:\s*(?:财年|FY))?", re.I)

_SCALE = {
    "百": 1e2, "千": 1e3, "万": 1e4, "亿": 1e8,
    "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12,
}
_CURRENCY = {
    "$": "USD", "us$": "USD", "usd": "USD",
    "¥": "CNY", "cny": "CNY", "rmb": "CNY",
    "hk$": "HKD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
    "jpy": "JPY",
    "人民币": "CNY", "美元": "USD", "欧元": "EUR", "港元": "HKD",
}
_MEASURE_UNIT_RE = re.compile(
    r"^\s*(?P<unit>千克|公斤|kg|克|g|吨|公里|千米|km|米|m|°\s*[Cc]|℃)(?![A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericFact:
    value: float
    kind: str = "number"
    unit: str = ""
    raw: str = ""
    entity: tuple[str, ...] = ()


class NumericVerificationSkill:
    """Execute a versioned YAML skill using a fixed primitive allowlist.

    The skill is deliberately non-agentic: YAML controls only thresholds,
    primitive order and conflict/fallback choices.  It cannot import code,
    invoke tools, or access the network.
    """

    def __init__(self, spec: Mapping[str, Any], *, artifact: Mapping[str, Any] | None = None) -> None:
        self.spec = dict(spec)
        self.artifact = dict(artifact or {})
        primitives = self.spec.get("primitives", self.spec.get("steps", []))
        normalized: list[str] = []
        for item in primitives:
            name = item.get("action", "") if isinstance(item, Mapping) else str(item)
            if name not in ALLOWED_PRIMITIVES:
                raise ValueError(f"unsupported numeric skill primitive: {name}")
            normalized.append(name)
        if not normalized:
            raise ValueError("numeric skill requires at least one primitive")
        self.primitives = tuple(normalized)
        enabled_composable = COMPOSABLE_PRIMITIVES & set(self.primitives)
        if enabled_composable:
            if "compare_value_unit_date" not in self.primitives:
                raise ValueError("composable numeric primitives require compare_value_unit_date")
            compare_index = self.primitives.index("compare_value_unit_date")
            if any(self.primitives.index(name) > compare_index for name in enabled_composable):
                raise ValueError("composable numeric primitives must run before compare_value_unit_date")
        # Shipped V1 artifacts call these scalar controls ``parameters``;
        # accept ``settings`` as a backwards-compatible alias for test/local
        # specs while keeping the registry schema strict.
        settings = dict(self.spec.get("parameters", self.spec.get("settings", {})))
        self.relative_tolerance = float(settings.get("relative_tolerance", 0.005))
        self.absolute_tolerance = float(settings.get("absolute_tolerance", 1e-9))
        self.date_tolerance_days = int(settings.get("date_tolerance_days", 0))
        if not 0 <= self.relative_tolerance <= 0.25:
            raise ValueError("relative_tolerance must be in [0, 0.25]")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative")
        if not 0 <= self.date_tolerance_days <= 366:
            raise ValueError("date_tolerance_days must be in [0, 366]")
        conflict_raw = self.spec.get("conflict_policy", settings.get("conflict_policy", "unknown"))
        self.conflict_policy = {
            "mark_unknown": "unknown",
            "mark_contradicted": "contradicted",
        }.get(str(conflict_raw), str(conflict_raw))
        if self.conflict_policy not in {"unknown", "contradicted"}:
            raise ValueError("conflict_policy must be unknown or contradicted")
        fallback_raw = self.spec.get("fallback", settings.get("fallback", "unknown"))
        if isinstance(fallback_raw, Mapping):
            fallback_values = set(str(value) for value in fallback_raw.values())
            self.fallback = "unknown" if fallback_values <= {"mark_unknown", "unknown"} else "invalid"
        else:
            self.fallback = {"mark_unknown": "unknown"}.get(str(fallback_raw), str(fallback_raw))
        if self.fallback != "unknown":
            raise ValueError("numeric skill fallback must be unknown")
        self.source_priority = tuple(self.spec.get("source_priority", []))
        limits = dict(self.spec.get("limits", {}))
        self.max_sources = max(1, int(limits.get("max_sources", 5)))
        self.max_source_spans = max(1, int(limits.get("max_source_spans", 10)))

    def __call__(self, claim: Claim, evidence: Sequence[Evidence]) -> dict[str, Any]:
        spans = [item for item in evidence if str(item.source_span or "").strip()]
        if "rank_primary_sources" in self.primitives and self.source_priority:
            ranks = {name: index for index, name in enumerate(self.source_priority)}
            spans.sort(key=lambda item: ranks.get(str(item.metadata.get("source_tier", "other")), len(ranks)))
        spans = spans[: min(self.max_sources, self.max_source_spans)]
        if not spans:
            return self._result(VerificationStatus.UNKNOWN, 0.0, evidence, "numeric_skill:no_source_span")

        claim_facts = self.extract_facts(claim.text)
        if not claim_facts:
            return self._result(VerificationStatus.UNKNOWN, 0.0, evidence, "numeric_skill:no_numeric_claim")

        enhanced = bool(COMPOSABLE_PRIMITIVES & set(self.primitives))
        verdicts: list[str] = []
        for source in spans:
            source_facts = self.extract_facts(source.source_span)
            if not source_facts:
                verdicts.append("unknown")
                continue
            if enhanced:
                verdicts.append(self._compare_source(claim_facts, source_facts))
            else:
                matched = all(any(self._matches(expected, observed) for observed in source_facts) for expected in claim_facts)
                verdicts.append("supported" if matched else "contradicted")

        has_support = "supported" in verdicts
        has_conflict = "contradicted" in verdicts
        if has_support and has_conflict:
            status = VerificationStatus(self.conflict_policy)
            return self._result(status, 0.65, spans, "numeric_skill:conflicting_source_values")
        if has_support:
            return self._result(VerificationStatus.SUPPORTED, 0.9, spans, "numeric_skill:value_unit_date_match")
        if has_conflict:
            return self._result(VerificationStatus.CONTRADICTED, 0.9, spans, "numeric_skill:value_unit_or_date_mismatch")
        return self._result(VerificationStatus.UNKNOWN, 0.2, spans, "numeric_skill:insufficient_numeric_evidence")

    @classmethod
    def extract_facts(cls, text: str) -> list[NumericFact]:
        facts: list[NumericFact] = []
        citation_spans = [match.span() for match in re.finditer(r"\[\d+\]", text)]
        year_spans: set[tuple[int, int]] = set()
        for match in _YEAR_RE.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in citation_spans):
                continue
            raw = match.group(0)
            fiscal = bool(re.search(r"FY|财年", raw, re.I))
            # A bare four-digit count can fall inside the calendar-year range
            # (for example, "2048 个样本").  Treat it as a date only when the
            # surrounding text supplies an explicit year/fiscal marker.
            suffix = text[match.end(): min(len(text), match.end() + 4)]
            if not fiscal and not re.match(r"\s*(?:年|年度)", suffix):
                continue
            year_spans.add(match.span())
            facts.append(NumericFact(
                float(match.group("year")),
                "fiscal_year" if fiscal else "date",
                "fiscal_year" if fiscal else "year",
                raw,
                entity_signature(text, match.start(), match.end()),
            ))
        for match in _NUMERIC_RE.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in citation_spans):
                continue
            if any(match.start() < end and match.end() > start for start, end in year_spans):
                continue
            raw_value = match.group("value")
            try:
                value = float(raw_value.replace(",", ""))
            except ValueError:
                continue
            scale_token = match.group("scale") or ""
            inline_measure_unit = ""
            # A separated lowercase ``m`` is metres, while compact/uppercase
            # ``M`` remains the conventional million scale abbreviation.
            if scale_token == "m" and re.search(r"\d\s+m\s*$", match.group(0)):
                inline_measure_unit = "m"
                scale_token = ""
            scale_raw = scale_token.lower()
            value *= _SCALE.get(scale_raw, 1.0)
            percent = bool(match.group("percent"))
            currency_raw = (match.group("currency") or "").lower()
            window = text[max(0, match.start() - 12): min(len(text), match.end() + 12)].lower()
            if not currency_raw:
                currency_raw = next((token for token in _CURRENCY if token in window), "")
            unit_match = _MEASURE_UNIT_RE.match(text[match.end():])
            measure_unit = inline_measure_unit
            if unit_match:
                raw_unit = re.sub(r"\s+", "", unit_match.group("unit")).lower()
                unit_spec = MEASUREMENT_UNITS.get(raw_unit)
                measure_unit = unit_spec[1] if unit_spec else raw_unit
                # Preserve the original allow-listed unit so conversion can
                # distinguish grams/metres from their canonical units.
                if raw_unit in {"克", "g", "米", "m"}:
                    measure_unit = raw_unit
            entity = entity_signature(text, match.start(), match.end())
            if percent:
                facts.append(NumericFact(value, "percentage", "%", match.group(0), entity))
            elif currency_raw:
                facts.append(NumericFact(value, "currency", _CURRENCY.get(currency_raw, currency_raw.upper()), match.group(0), entity))
            elif measure_unit:
                facts.append(NumericFact(value, "measurement", measure_unit, match.group(0), entity))
            else:
                facts.append(NumericFact(value, "number", scale_raw, match.group(0), entity))
        return facts

    def _compare_source(
        self,
        expected_facts: Sequence[NumericFact],
        observed_facts: Sequence[NumericFact],
    ) -> str:
        """Compare one source using only explicitly enabled safe primitives."""

        align = "align_entity_value" in self.primitives
        filter_irrelevant = "filter_semantically_irrelevant_numbers" in self.primitives
        convert_units = "convert_measurement_unit" in self.primitives
        saw_uncertain = False
        for expected in expected_facts:
            candidates = list(observed_facts)
            if (align or filter_irrelevant) and expected.entity:
                candidates = [
                    observed
                    for observed in candidates
                    if (
                        entities_compatible(expected.entity, observed.entity)
                        if align
                        else concepts_compatible(expected.entity, observed.entity)
                    )
                ]
                if not candidates:
                    return "unknown"
            if any(self._matches(expected, observed, convert_units=convert_units) for observed in candidates):
                continue
            if convert_units and expected.kind == "measurement":
                comparable = [
                    observed
                    for observed in candidates
                    if observed.kind == "measurement"
                    and canonical_measurement(expected.value, expected.unit) is not None
                    and canonical_measurement(observed.value, observed.unit) is not None
                ]
                if not comparable:
                    saw_uncertain = True
                    continue
            return "contradicted"
        return "unknown" if saw_uncertain else "supported"

    def _matches(
        self,
        expected: NumericFact,
        observed: NumericFact,
        *,
        convert_units: bool = False,
    ) -> bool:
        if expected.kind != observed.kind:
            return False
        if convert_units and expected.kind == "measurement":
            left = canonical_measurement(expected.value, expected.unit)
            right = canonical_measurement(observed.value, observed.unit)
            if left is None or right is None or left[:2] != right[:2]:
                return False
            tolerance = max(self.absolute_tolerance, abs(left[2]) * self.relative_tolerance)
            return abs(left[2] - right[2]) <= tolerance
        if expected.unit and observed.unit and expected.unit != observed.unit:
            return False
        if expected.kind in {"date", "fiscal_year"}:
            tolerance_years = self.date_tolerance_days / 365.0
            return abs(expected.value - observed.value) <= tolerance_years
        tolerance = max(self.absolute_tolerance, abs(expected.value) * self.relative_tolerance)
        return abs(expected.value - observed.value) <= tolerance

    def _result(
        self,
        status: VerificationStatus,
        confidence: float,
        evidence: Sequence[Evidence],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "status": status.value,
            "confidence": confidence,
            "evidence": [item.to_dict() for item in evidence],
            "reason": reason,
            "skill": dict(self.artifact),
            "primitive_trace": [
                name for name in self.primitives if name in COMPOSABLE_PRIMITIVES
            ],
        }
