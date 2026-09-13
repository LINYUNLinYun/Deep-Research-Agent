"""Safe runtime for the versioned ``search_control_policy`` YAML artifact.

The runtime intentionally supports a very small declarative language.  Policy
files select an action from an allowlist and never evaluate Python expressions
or import functions named by configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import SchemaValidationError, validate_search_policy


class PolicyValidationError(ValueError):
    """Raised when a policy contains unsupported or unsafe configuration."""


ALLOWED_SIGNALS = {
    "result_count",
    "duplicate_ratio",
    "evidence_novelty",
    "unresolved_claims",
    "search_attempts",
    "remaining_search_budget",
    "provider_health",
    "query_type",
}
ALLOWED_ACTIONS = {
    "accept_results",
    "rewrite_uncovered_facets",
    "switch_provider",
    "invoke_numeric_verification",
    "stop_search",
}
ALLOWED_OPERATORS = {">", ">=", "<", "<=", "==", "in"}
_TOP_LEVEL = {
    "artifact_id", "version", "parent", "description", "decision_point",
    "parameters", "rules", "fallback", "hard_limits",
}
_RULE_FIELDS = {"id", "priority", "all", "any", "not", "action", "limits"}
_CONDITION_FIELDS = {"signal", "operator", "value"}
_ACTION_FIELDS = {"type", "skill_id"}
_LIMIT_FIELDS = {"max_activations_per_run"}


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    rule_id: str
    signals: dict[str, Any]
    artifact_id: str
    version: str
    sha256: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_point": "after_search",
            "action": self.action,
            "rule_id": self.rule_id,
            "signals": dict(self.signals),
            "artifact": {
                "id": self.artifact_id,
                "version": self.version,
                "sha256": self.sha256,
            },
            "reason": self.reason,
        }


class SearchControlPolicy:
    """Validated and immutable search-policy evaluator."""

    def __init__(self, spec: Mapping[str, Any], *, sha256: str = "") -> None:
        self.spec = _validate_policy(dict(spec))
        self.artifact_id = str(self.spec.get("artifact_id"))
        self.version = str(self.spec.get("version", "unversioned"))
        self.sha256 = str(sha256)
        parameters = self.spec["parameters"]
        self.novelty_threshold = float(parameters["novelty_threshold"])
        self.query_similarity_threshold = float(parameters["query_similarity_threshold"])
        self.max_rewrites = int(parameters["max_rewrites"])
        hard_limits = self.spec.get("hard_limits", {})
        self.max_search_attempts = int(hard_limits.get("max_search_attempts", 100))
        self.max_total_rewrites = int(hard_limits.get("max_total_rewrites", self.max_rewrites))
        self._activations: dict[str, int] = {}
        self._total_rewrites = 0

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any], *, sha256: str = "") -> "SearchControlPolicy":
        return cls(spec, sha256=sha256)

    @classmethod
    def from_file(cls, path: str | Path, *, sha256: str = "") -> "SearchControlPolicy":
        with Path(path).open("r", encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        if not isinstance(spec, Mapping):
            raise PolicyValidationError("policy document must be a mapping")
        return cls(spec, sha256=sha256)

    def reset(self) -> None:
        self._activations.clear()
        self._total_rewrites = 0

    def decide(self, signals: Mapping[str, Any]) -> PolicyDecision:
        clean = {name: signals.get(name) for name in ALLOWED_SIGNALS}
        if int(clean.get("search_attempts") or 0) >= self.max_search_attempts:
            return self._decision("stop_search", "hard_limit:max_search_attempts", clean, "search hard cap reached")
        for rule in sorted(self.spec.get("rules", []), key=lambda item: -item["priority"]):
            rule_id = rule["id"]
            limit = rule.get("limits", {}).get("max_activations_per_run")
            if limit is not None and self._activations.get(rule_id, 0) >= limit:
                continue
            if _matches(rule, clean):
                action = rule["action"]["type"]
                if action == "rewrite_uncovered_facets" and self._total_rewrites >= self.max_total_rewrites:
                    continue
                self._activations[rule_id] = self._activations.get(rule_id, 0) + 1
                if action == "rewrite_uncovered_facets":
                    self._total_rewrites += 1
                return self._decision(action, rule_id, clean, "rule matched")
        fallback = self.spec.get("fallback", {"action": {"type": "accept_results"}})
        return self._decision(fallback["action"]["type"], "fallback", clean, "no rule matched")

    def metadata(self) -> dict[str, str]:
        return {"id": self.artifact_id, "version": self.version, "sha256": self.sha256}

    def _decision(self, action: str, rule_id: str, signals: dict[str, Any], reason: str) -> PolicyDecision:
        return PolicyDecision(action, rule_id, signals, self.artifact_id, self.version, self.sha256, reason)


def _validate_policy(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_search_policy(spec)
    except SchemaValidationError as exc:
        raise PolicyValidationError(str(exc)) from exc
    return spec


def _bounded_float(spec: Mapping[str, Any], field: str, low: float, high: float) -> None:
    if field not in spec:
        return
    value = spec[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= float(value) <= high:
        raise PolicyValidationError(f"{field} must be numeric in [{low}, {high}]")


def _validate_condition(condition: Any) -> None:
    if not isinstance(condition, dict) or set(condition) != _CONDITION_FIELDS:
        raise PolicyValidationError("condition must contain signal/operator/value")
    if condition["signal"] not in ALLOWED_SIGNALS:
        raise PolicyValidationError(f"unknown signal: {condition['signal']}")
    if condition["operator"] not in ALLOWED_OPERATORS:
        raise PolicyValidationError(f"unknown operator: {condition['operator']}")
    if condition["operator"] == "in" and not isinstance(condition["value"], (list, tuple, set)):
        raise PolicyValidationError("the 'in' operator requires a collection value")


def _validate_action(action: Any) -> None:
    if not isinstance(action, dict) or set(action) - _ACTION_FIELDS or "type" not in action:
        raise PolicyValidationError("action must contain only type/skill_id")
    if action["type"] not in ALLOWED_ACTIONS:
        raise PolicyValidationError(f"unknown action: {action['type']}")
    if "skill_id" in action and not isinstance(action["skill_id"], str):
        raise PolicyValidationError("skill_id must be a string")


def _matches(rule: Mapping[str, Any], signals: Mapping[str, Any]) -> bool:
    group = next(key for key in ("all", "any") if key in rule)
    values = [_compare(signals.get(c["signal"]), c["operator"], c["value"]) for c in rule[group]]
    if group == "all":
        return all(values)
    return any(values)


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    try:
        if operator == "==":
            return actual == expected
        if operator == "in":
            return actual in expected
        if operator == ">":
            return actual > expected
        if operator == ">=":
            return actual >= expected
        if operator == "<":
            return actual < expected
        if operator == "<=":
            return actual <= expected
    except (TypeError, ValueError):
        return False
    return False
