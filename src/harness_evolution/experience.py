"""Append-only Harness experiences and deterministic failure attribution."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class ExperienceRecord:
    run_id: str
    query_id: str
    query_type: str = ""
    query_features: dict[str, Any] = field(default_factory=dict)
    policy_versions: dict[str, Any] = field(default_factory=dict)
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    tool_attempts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    failure_labels: list[str] = field(default_factory=list)
    executor_model: str = ""
    dataset_split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperienceStore:
    """Minimal append-only JSONL store, intentionally separate from GRPO memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ExperienceRecord | Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict() if isinstance(record, ExperienceRecord) else dict(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FailureMiner:
    """Assign reproducible labels before an optional LLM explains them."""

    def label(self, record: Mapping[str, Any]) -> list[str]:
        telemetry = dict(record.get("harness", record.get("telemetry", {})) or {})
        search = dict(telemetry.get("search", {}) or {})
        stats = dict(search.get("stats", {}) or {})
        verification = dict(telemetry.get("evidence_verification", {}) or {})
        skill_info = dict(telemetry.get("skills", {}).get("verify_numeric_claim_skill", {}) or {})
        decisions = list(telemetry.get("decision_trace", record.get("decisions", [])) or [])
        policy_decisions = list(search.get("policy_decisions", []) or [])
        if not policy_decisions:
            policy_decisions = [
                item for item in decisions
                if isinstance(item, Mapping)
                and isinstance(item.get("signals"), Mapping)
                and isinstance(item["signals"].get("policy_signals"), Mapping)
            ]
        labels: set[str] = set()

        expected = str(record.get("expected", "") or "").lower()
        predicted = str(record.get("predicted", "") or "").lower()
        if expected and predicted and expected != predicted:
            labels.add("numeric_verdict_mismatch")
            primitive_gap_labels = {
                "irrelevant_number": "missing_irrelevant_number_filter",
                "multi_entity_alignment": "missing_entity_value_alignment",
                "unit_missing_or_conversion": "missing_measurement_unit_conversion",
            }
            gap_label = primitive_gap_labels.get(str(record.get("query_type", "")))
            if gap_label:
                labels.add(gap_label)
            if expected == "contradicted" and predicted == "supported":
                labels.add("false_supported_numeric_claim")
            elif expected == "unknown" and predicted == "contradicted":
                labels.add("irrelevant_numeric_contradiction")
            elif expected == "supported" and predicted != "supported":
                labels.add("missed_numeric_equivalence")
        if record.get("normalization_correct") is False:
            labels.add("numeric_normalization_failure")

        calls = max(int(stats.get("calls", 0) or 0), 1)
        duplicates = int(stats.get("duplicate_results", 0) or 0)
        new_results = int(stats.get("new_results", 0) or 0)
        rewrites = int(stats.get("rewritten_queries", 0) or 0)
        if duplicates / max(duplicates + new_results, 1) > 0.4:
            labels.add("duplicate_search")
        if new_results / calls < 0.25 and int(stats.get("backend_calls", 0) or 0) > 0:
            labels.add("low_evidence_novelty")
        if new_results == 0 and rewrites == 0 and int(stats.get("backend_calls", 0) or 0) > 0:
            labels.add("missed_query_rewrite")

        decision_signals: list[tuple[Mapping[str, Any], str, str]] = []
        for item in policy_decisions:
            if not isinstance(item, Mapping):
                continue
            wrapper = item.get("signals", {})
            if not isinstance(wrapper, Mapping):
                wrapper = {}
            nested = wrapper.get("policy_signals", wrapper)
            if not isinstance(nested, Mapping):
                continue
            action = str(item.get("action", ""))
            rule_id = str(wrapper.get("rule_id", item.get("rule_id", "")))
            decision_signals.append((nested, action, rule_id))

        if decision_signals:
            if all(int(signals.get("unresolved_claims", 0) or 0) == 0 for signals, _, _ in decision_signals):
                labels.add("unresolved_signal_unavailable")
            if any(float(signals.get("duplicate_ratio", 0.0) or 0.0) >= 0.4 for signals, _, _ in decision_signals):
                labels.add("duplicate_search")
            if any(
                int(signals.get("result_count", 0) or 0) > 0
                and float(signals.get("evidence_novelty", 1.0) or 0.0) < 0.25
                for signals, _, _ in decision_signals
            ):
                labels.add("low_evidence_novelty")
            if any(rule_id == "hard_limit:max_search_attempts" for _, _, rule_id in decision_signals):
                labels.add("search_hard_cap_reached")
            if any(
                rule_id == "hard_limit:max_search_attempts"
                and int(signals.get("result_count", 0) or 0) == 0
                and int(signals.get("remaining_search_budget", 0) or 0) > 0
                for signals, _, rule_id in decision_signals
            ):
                labels.add("rewrite_blocked_by_hard_cap")
            if any(
                action == "accept_results"
                and int(signals.get("remaining_search_budget", 0) or 0) > 0
                and (
                    int(signals.get("result_count", 0) or 0) == 0
                    or float(signals.get("evidence_novelty", 1.0) or 0.0) < 0.25
                )
                for signals, action, _ in decision_signals
            ):
                labels.add("missed_query_rewrite")
        if verification.get("contradicted", 0):
            labels.add("contradicted_claim")
        if verification.get("unknown", 0):
            labels.add("unsupported_claim")
        if verification.get("total_claims", 0) and not skill_info.get("triggered") and not record.get("selected_skills"):
            labels.add("missed_skill_activation")

        for item in decisions:
            action = str(item.get("action", ""))
            signals = dict(item.get("signals", {}) or {})
            if action in {"synthesize", "stop_search"} and signals.get("unresolved_claims"):
                labels.add("premature_stopping")
            if action == "rewrite_uncovered_facets" and not signals.get("rewrite_success", True):
                labels.add("skill_execution_failure")
        return sorted(labels)

    def mine(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for value in records:
            item = dict(value)
            item["failure_labels"] = self.label(item)
            output.append(item)
        return output
