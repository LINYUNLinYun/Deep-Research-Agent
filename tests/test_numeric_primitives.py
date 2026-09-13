"""Contract tests for the first safe Numeric Verification primitives.

These tests intentionally exercise the public ``NumericVerificationSkill``
callable rather than private parser helpers.  The primitive names are part of
the versioned Skill allow-list; their implementations may evolve, but the
conservative verdict contract should remain stable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from src.evidence.schemas import Claim, Evidence, VerificationStatus
from src.harness_evolution.numeric_skill import NumericVerificationSkill


_BASE_PRIMITIVES = [
    "detect_numeric_claim",
    "normalize_number",
    "normalize_percentage",
    "normalize_currency_unit",
    "normalize_date_or_fiscal_year",
    "compare_value_unit_date",
    "resolve_source_conflict",
    "classify_supported_contradicted_unknown",
]

_SAFE_PRIMITIVES = [
    "align_entity_value",
    "convert_measurement_unit",
    "filter_semantically_irrelevant_numbers",
]


def _spec(*extra: str) -> dict:
    """Build a local Skill spec without touching a versioned YAML artifact."""

    primitives = list(_BASE_PRIMITIVES)
    insertion_point = primitives.index("compare_value_unit_date")
    primitives[insertion_point:insertion_point] = extra
    return {
        "primitives": primitives,
        "parameters": {
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.001,
            "date_tolerance_days": 0,
        },
        "source_priority": ["primary", "official", "academic", "secondary", "other"],
        "conflict_policy": "mark_unknown",
        "fallback": {"no_source": "mark_unknown", "execution_error": "mark_unknown"},
        "limits": {"max_sources": 5, "max_source_spans": 10},
    }


def _status(result: object) -> str:
    """Accept the current dict API and the serialisable result enum API."""

    if isinstance(result, Mapping):
        return str(result["status"])
    status = getattr(result, "status")
    return status.value if isinstance(status, VerificationStatus) else str(status)


def _verify(
    spec: Mapping[str, object],
    claim_text: str,
    source_spans: Sequence[str],
) -> str:
    skill = NumericVerificationSkill(spec)
    claim = Claim("primitive-contract", claim_text)
    evidence = [Evidence(source_span=span) for span in source_spans]
    return _status(skill(claim, evidence))


def test_align_entity_value_rejects_swapped_a_b_values() -> None:
    """Matching a set of numbers is insufficient when entities are swapped."""

    status = _verify(
        _spec("align_entity_value"),
        "A产品收入为 100 万元，B产品收入为 80 万元。",
        ["A产品收入为 80 万元，B产品收入为 100 万元。"],
    )
    assert status == VerificationStatus.CONTRADICTED.value


def test_align_entity_value_keeps_entity_aligned_values_supported() -> None:
    status = _verify(
        _spec("align_entity_value"),
        "A产品收入为 100 万元，B产品收入为 80 万元。",
        ["A产品收入为 100 万元，B产品收入为 80 万元。"],
    )
    assert status == VerificationStatus.SUPPORTED.value


@pytest.mark.parametrize(
    ("claim", "source"),
    [
        ("设备重量为 2.5 kg。", "设备重量为 2500 g。"),
        ("路线长度为 1 km。", "路线长度为 1000 m。"),
    ],
)
def test_convert_measurement_unit_supports_whitelisted_metric_equivalences(
    claim: str,
    source: str,
) -> None:
    assert _verify(_spec("convert_measurement_unit"), claim, [source]) == (
        VerificationStatus.SUPPORTED.value
    )


def test_convert_measurement_unit_unknown_unit_is_conservative() -> None:
    """An unrecognised conversion must not become a false supported verdict."""

    status = _verify(
        _spec("convert_measurement_unit"),
        "设备重量为 2.5 kg。",
        ["设备重量为 2.5 lb。"],
    )
    assert status == VerificationStatus.UNKNOWN.value


def test_filter_irrelevant_numbers_does_not_contradict_missing_metric() -> None:
    """A sample size cannot refute a claim about an unreported rate."""

    status = _verify(
        _spec("filter_semantically_irrelevant_numbers"),
        "该试验住院率下降 20%。",
        ["研究纳入 120 名患者，随访 12 个月，但未报告住院率变化。"],
    )
    assert status == VerificationStatus.UNKNOWN.value


def test_filter_irrelevant_numbers_still_contradicts_same_entity_wrong_value() -> None:
    status = _verify(
        _spec("filter_semantically_irrelevant_numbers"),
        "该试验住院率下降 20%。",
        ["该试验住院率下降 30%。"],
    )
    assert status == VerificationStatus.CONTRADICTED.value


def test_entity_ambiguity_is_conservative_unknown() -> None:
    """A claim without an entity cannot choose between multiple source entities."""

    status = _verify(
        _spec("align_entity_value"),
        "收入为 100 万元。",
        ["A产品收入为 100 万元，B产品收入为 80 万元。"],
    )
    assert status == VerificationStatus.UNKNOWN.value


def test_new_primitives_still_require_an_attributable_source_span() -> None:
    skill = NumericVerificationSkill(
        _spec("align_entity_value", "convert_measurement_unit", "filter_semantically_irrelevant_numbers")
    )
    result = skill(
        Claim("no-span", "设备重量为 2.5 kg。"),
        [Evidence(source_url="https://example.test/weight")],
    )
    assert _status(result) == VerificationStatus.UNKNOWN.value


def test_disabling_new_primitives_preserves_legacy_numeric_only_behavior() -> None:
    """Production v0001 semantics remain unchanged unless YAML opts in."""

    status = _verify(
        _spec(),
        "A产品收入为 100 万元，B产品收入为 80 万元。",
        ["A产品收入为 80 万元，B产品收入为 100 万元。"],
    )
    # The legacy implementation compares an unordered set of numeric facts;
    # this deliberately documents the pre-primitive baseline for ablations.
    assert status == VerificationStatus.SUPPORTED.value


def test_composable_primitive_after_compare_is_rejected() -> None:
    spec = _spec()
    classifier_index = spec["primitives"].index(
        "classify_supported_contradicted_unknown"
    )
    spec["primitives"].insert(classifier_index, "align_entity_value")

    with pytest.raises(ValueError, match="before compare_value_unit_date"):
        NumericVerificationSkill(spec)


def test_meter_unit_is_distinct_from_uppercase_million_scale() -> None:
    metre = NumericVerificationSkill.extract_facts("路线长度为 1000 m。")
    million = NumericVerificationSkill.extract_facts("平台服务 2.5M users。")

    assert [(fact.kind, fact.value, fact.unit) for fact in metre] == [
        ("measurement", 1000.0, "m")
    ]
    assert [(fact.kind, fact.value) for fact in million] == [
        ("number", 2_500_000.0)
    ]
