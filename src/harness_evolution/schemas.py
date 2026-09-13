"""Allow-list validators for evolvable Harness YAML artifacts.

The validators deliberately implement a small DSL.  Artifact files are data,
not executable configuration: arbitrary callables, expressions and unknown
fields are rejected before a version can enter a registry.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when an evolvable artifact is outside the V1 DSL."""


POLICY_SIGNALS = {
    "result_count",
    "duplicate_ratio",
    "evidence_novelty",
    "unresolved_claims",
    "search_attempts",
    "remaining_search_budget",
    "provider_health",
    "query_type",
}
POLICY_ACTIONS = {
    "accept_results",
    "rewrite_uncovered_facets",
    "switch_provider",
    "invoke_numeric_verification",
    "stop_search",
}
POLICY_OPERATORS = {">", ">=", "<", "<=", "==", "in"}
SKILL_PRIMITIVES = {
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{name} must be a mapping")
    return value


def _only(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise SchemaValidationError(f"unknown {name} fields: {sorted(unknown)}")


def _number(value: Any, name: str, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be numeric")
    if not low <= float(value) <= high:
        raise SchemaValidationError(f"{name} must be in [{low}, {high}]")


def _positive_int(value: Any, name: str, maximum: int = 100) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise SchemaValidationError(f"{name} must be an integer in [0, {maximum}]")


def _condition(value: Any, name: str) -> None:
    condition = _mapping(value, name)
    _only(condition, {"signal", "operator", "value"}, name)
    if condition.get("signal") not in POLICY_SIGNALS:
        raise SchemaValidationError(f"unknown policy signal: {condition.get('signal')!r}")
    if condition.get("operator") not in POLICY_OPERATORS:
        raise SchemaValidationError(f"unknown policy operator: {condition.get('operator')!r}")
    if "value" not in condition:
        raise SchemaValidationError(f"{name}.value is required")


def _action(value: Any, name: str) -> None:
    action = _mapping(value, name)
    _only(action, {"type"}, name)
    if action.get("type") not in POLICY_ACTIONS:
        raise SchemaValidationError(f"unknown policy action: {action.get('type')!r}")


def validate_search_policy(data: Mapping[str, Any]) -> None:
    """Validate one ``search_control_policy`` artifact."""

    data = _mapping(data, "policy")
    _only(
        data,
        {
            "artifact_id",
            "version",
            "parent",
            "description",
            "decision_point",
            "parameters",
            "rules",
            "fallback",
            "hard_limits",
        },
        "policy",
    )
    if data.get("artifact_id") != "search_control_policy":
        raise SchemaValidationError("policy artifact_id must be search_control_policy")
    if data.get("decision_point") != "after_search":
        raise SchemaValidationError("V1 policy decision_point must be after_search")

    params = _mapping(data.get("parameters", {}), "parameters")
    _only(params, {"novelty_threshold", "query_similarity_threshold", "max_rewrites"}, "parameters")
    _number(params.get("novelty_threshold"), "novelty_threshold", 0.0, 1.0)
    _number(params.get("query_similarity_threshold"), "query_similarity_threshold", 0.0, 1.0)
    _positive_int(params.get("max_rewrites"), "max_rewrites", 10)

    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise SchemaValidationError("rules must be a non-empty list")
    seen: set[str] = set()
    for index, raw_rule in enumerate(rules):
        rule = _mapping(raw_rule, f"rules[{index}]")
        _only(rule, {"id", "priority", "all", "any", "action", "limits"}, f"rules[{index}]")
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
            raise SchemaValidationError("rule ids must be non-empty and unique")
        seen.add(rule_id)
        _positive_int(rule.get("priority"), f"rules[{index}].priority", 10000)
        groups = [key for key in ("all", "any") if key in rule]
        if len(groups) != 1:
            raise SchemaValidationError(f"rules[{index}] must define exactly one of all/any")
        conditions = rule[groups[0]]
        if not isinstance(conditions, list) or not conditions:
            raise SchemaValidationError(f"rules[{index}].{groups[0]} must be a non-empty list")
        for cond_index, condition in enumerate(conditions):
            _condition(condition, f"rules[{index}].{groups[0]}[{cond_index}]")
        _action(rule.get("action"), f"rules[{index}].action")
        limits = _mapping(rule.get("limits", {}), f"rules[{index}].limits")
        _only(limits, {"max_activations_per_run"}, f"rules[{index}].limits")
        _positive_int(limits.get("max_activations_per_run", 1), "max_activations_per_run", 10)

    fallback = _mapping(data.get("fallback"), "fallback")
    _only(fallback, {"action"}, "fallback")
    _action(fallback.get("action"), "fallback.action")
    hard_limits = _mapping(data.get("hard_limits", {}), "hard_limits")
    _only(hard_limits, {"max_search_attempts", "max_total_rewrites"}, "hard_limits")
    for key, value in hard_limits.items():
        _positive_int(value, key, 100)


def validate_numeric_skill(data: Mapping[str, Any]) -> None:
    """Validate one ``verify_numeric_claim_skill`` artifact."""

    data = _mapping(data, "skill")
    _only(
        data,
        {
            "artifact_id",
            "version",
            "parent",
            "description",
            "claim_types",
            "primitives",
            "parameters",
            "source_priority",
            "conflict_policy",
            "fallback",
            "limits",
        },
        "skill",
    )
    if data.get("artifact_id") != "verify_numeric_claim_skill":
        raise SchemaValidationError("skill artifact_id must be verify_numeric_claim_skill")
    allowed_claim_types = {"number", "percentage", "currency", "date", "fiscal_year"}
    claim_types = data.get("claim_types")
    if not isinstance(claim_types, list) or not claim_types or not set(claim_types) <= allowed_claim_types:
        raise SchemaValidationError("claim_types contains an unsupported type")
    primitives = data.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise SchemaValidationError("primitives must be a non-empty list")
    if len(primitives) != len(set(primitives)) or not set(primitives) <= SKILL_PRIMITIVES:
        raise SchemaValidationError("primitives must be unique allow-listed names")
    if primitives[-1] != "classify_supported_contradicted_unknown":
        raise SchemaValidationError("the final primitive must classify the result")
    composable = {
        "align_entity_value",
        "convert_measurement_unit",
        "filter_semantically_irrelevant_numbers",
    } & set(primitives)
    if composable:
        if "compare_value_unit_date" not in primitives:
            raise SchemaValidationError("composable primitives require compare_value_unit_date")
        compare_index = primitives.index("compare_value_unit_date")
        if any(primitives.index(name) > compare_index for name in composable):
            raise SchemaValidationError("composable primitives must precede compare_value_unit_date")

    params = _mapping(data.get("parameters", {}), "parameters")
    _only(params, {"absolute_tolerance", "relative_tolerance", "date_tolerance_days"}, "parameters")
    _number(params.get("absolute_tolerance"), "absolute_tolerance", 0.0, 1_000_000.0)
    _number(params.get("relative_tolerance"), "relative_tolerance", 0.0, 1.0)
    _positive_int(params.get("date_tolerance_days"), "date_tolerance_days", 366)
    priorities = data.get("source_priority")
    allowed_sources = {"primary", "official", "academic", "secondary", "other"}
    if not isinstance(priorities, list) or not priorities or len(priorities) != len(set(priorities)):
        raise SchemaValidationError("source_priority must be a non-empty unique list")
    if not set(priorities) <= allowed_sources:
        raise SchemaValidationError("source_priority contains an unsupported tier")
    if data.get("conflict_policy") not in {"mark_unknown", "mark_contradicted"}:
        raise SchemaValidationError("unsupported conflict_policy")
    fallback = _mapping(data.get("fallback"), "fallback")
    _only(fallback, {"no_source", "execution_error"}, "fallback")
    if set(fallback.values()) - {"mark_unknown"}:
        raise SchemaValidationError("V1 skill fallbacks must be mark_unknown")
    limits = _mapping(data.get("limits", {}), "limits")
    _only(limits, {"max_sources", "max_source_spans"}, "limits")
    for key, value in limits.items():
        _positive_int(value, key, 100)


def validate_artifact(data: Mapping[str, Any]) -> None:
    artifact_id = data.get("artifact_id") if isinstance(data, Mapping) else None
    if artifact_id == "search_control_policy":
        validate_search_policy(data)
    elif artifact_id == "verify_numeric_claim_skill":
        validate_numeric_skill(data)
    else:
        raise SchemaValidationError(f"unsupported artifact_id: {artifact_id!r}")
