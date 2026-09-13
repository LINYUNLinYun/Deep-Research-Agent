"""Bounded YAML candidate generation for the two V1 evolution tracks."""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class CandidatePatch:
    track: str
    field: str
    old: Any
    new: Any
    hypothesis: str
    patch_kind: str = "single_field"
    bundle_id: str | None = None
    added_primitives: tuple[str, ...] = ()


class CandidateGenerator:
    """Use an LLM for a hypothesis, then enumerate only safe one-field edits."""

    TRACKS = {"policy", "skill"}
    SAFE_PRIMITIVE_BUNDLE = "core_numeric_grounding_v1"
    SAFE_PRIMITIVE_BUNDLE_MEMBERS = (
        "align_entity_value",
        "convert_measurement_unit",
        "filter_semantically_irrelevant_numbers",
    )
    SAFE_SKILL_SUGGESTIONS = {
        *SAFE_PRIMITIVE_BUNDLE_MEMBERS,
        SAFE_PRIMITIVE_BUNDLE,
    }

    def __init__(self, proposer: Callable[..., Any] | None = None, max_candidates: int = 5) -> None:
        self.proposer = proposer
        self.max_candidates = min(max(int(max_candidates), 1), 5)
        self.proposal_telemetry: dict[str, Any] = {
            "requested": proposer is not None,
            "succeeded": False,
            "structured": False,
            "fallback_reason": "no_proposer" if proposer is None else "",
            "primitive_suggestions": [],
            "rejected_suggestions": [],
            "candidate_screening": [],
        }

    async def generate(
        self,
        track: str,
        current: Mapping[str, Any],
        experiences: Sequence[Mapping[str, Any]],
    ) -> list[CandidatePatch]:
        if track not in self.TRACKS:
            raise ValueError(f"unsupported evolution track: {track}")
        safe_experiences = [
            item
            for item in experiences
            if str(item.get("dataset_split", item.get("split", "")))
            .strip()
            .lower()
            .replace("-", "_")
            not in {"heldout", "held_out", "test"}
        ]
        hypothesis, primitive_suggestions = await self._proposal(
            track, current, safe_experiences
        )
        candidates = (
            self._policy_candidates(current, hypothesis, safe_experiences)
            if track == "policy"
            else self._skill_candidates(
                current,
                hypothesis,
                safe_experiences,
                primitive_suggestions=primitive_suggestions,
            )
        )
        if track == "policy":
            candidates = self._screen_policy_candidates(current, candidates, safe_experiences)
        return candidates[: self.max_candidates]

    def _screen_policy_candidates(
        self,
        current: Mapping[str, Any],
        candidates: Sequence[CandidatePatch],
        experiences: Sequence[Mapping[str, Any]],
    ) -> list[CandidatePatch]:
        """Drop policy candidates that are no-ops on recorded Miner signals."""

        if current.get("artifact_id") != "search_control_policy":
            return list(candidates)
        from .policy import SearchControlPolicy

        records_with_decisions = []
        for record in experiences:
            harness = record.get("harness", record.get("telemetry", {}))
            search = harness.get("search", {}) if isinstance(harness, Mapping) else {}
            decisions = search.get("policy_decisions", []) if isinstance(search, Mapping) else []
            if isinstance(decisions, list) and decisions:
                records_with_decisions.append(decisions)
        if not records_with_decisions:
            return list(candidates)

        retained: list[CandidatePatch] = []
        screening: list[dict[str, Any]] = []
        for patch in candidates:
            baseline = SearchControlPolicy(current)
            candidate = SearchControlPolicy(apply_candidate_patch(current, patch))
            changed = 0
            evaluated = 0
            for decisions in records_with_decisions:
                baseline.reset()
                candidate.reset()
                for item in decisions:
                    if not isinstance(item, Mapping):
                        continue
                    signals = item.get("signals", {})
                    if isinstance(signals, Mapping) and isinstance(signals.get("policy_signals"), Mapping):
                        signals = signals["policy_signals"]
                    if not isinstance(signals, Mapping):
                        continue
                    before = baseline.decide(signals)
                    after = candidate.decide(signals)
                    evaluated += 1
                    if (before.action, before.rule_id) != (after.action, after.rule_id):
                        changed += 1
            screening.append({
                "field": patch.field,
                "patch_kind": patch.patch_kind,
                "evaluated_decisions": evaluated,
                "changed_decisions": changed,
                "retained": changed > 0,
            })
            if changed > 0:
                retained.append(patch)
        self.proposal_telemetry["candidate_screening"] = screening
        return retained

    async def _proposal(
        self,
        track: str,
        current: Mapping[str, Any],
        experiences: Sequence[Mapping[str, Any]],
    ) -> tuple[str, tuple[str, ...]]:
        fallback = f"bounded {track} adjustment based on deterministic failure labels"
        if self.proposer is None:
            return fallback, ()
        safe_summary = [
            {
                "query_type": item.get("query_type", ""),
                "failure_labels": item.get("failure_labels", []),
                "final_metrics": item.get(
                    "final_metrics",
                    item.get("metrics", item.get("score_detail", {})),
                ),
            }
            for item in experiences[:50]
            if item.get("dataset_split", "") != "held_out"
        ]
        prompt = {
            "track": track,
            "current_yaml": dict(current),
            "development_summary": safe_summary,
            "available_safe_primitives": (
                sorted(self.SAFE_SKILL_SUGGESTIONS) if track == "skill" else []
            ),
            "instruction": (
                "Return JSON with a short hypothesis and primitive_suggestions ordered by expected "
                "Development benefit. Suggestions must come only from available_safe_primitives. "
                "Do not emit code, YAML, function paths, or access held-out data."
            ),
        }
        try:
            value = self.proposer([{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
            if inspect.isawaitable(value):
                value = await value
            if isinstance(value, Mapping):
                content = value.get("content", value)
            else:
                content = getattr(value, "content", value)
            if isinstance(content, str):
                if not content.strip():
                    self.proposal_telemetry["fallback_reason"] = "empty_response"
                    return fallback, ()
                try:
                    parsed = json.loads(content)
                    suggestions = parsed.get("primitive_suggestions", [])
                    if not isinstance(suggestions, list):
                        suggestions = []
                    raw_suggestions = tuple(dict.fromkeys(str(name) for name in suggestions))
                    safe_suggestions = tuple(
                        name for name in raw_suggestions if name in self.SAFE_SKILL_SUGGESTIONS
                    )
                    self.proposal_telemetry.update({
                        "succeeded": True,
                        "structured": True,
                        "fallback_reason": "",
                        "primitive_suggestions": list(safe_suggestions),
                        "rejected_suggestions": [
                            name for name in raw_suggestions if name not in self.SAFE_SKILL_SUGGESTIONS
                        ],
                    })
                    return str(parsed.get("hypothesis", fallback))[:500], safe_suggestions
                except json.JSONDecodeError:
                    self.proposal_telemetry.update({
                        "succeeded": True,
                        "structured": False,
                        "fallback_reason": "non_json_response",
                    })
                    return content.strip()[:500], ()
            self.proposal_telemetry["fallback_reason"] = "empty_response"
        except Exception as exc:
            self.proposal_telemetry["fallback_reason"] = f"proposer_error:{type(exc).__name__}"
        return fallback, ()

    @staticmethod
    def _policy_candidates(
        current: Mapping[str, Any],
        hypothesis: str,
        experiences: Sequence[Mapping[str, Any]] = (),
    ) -> list[CandidatePatch]:
        settings = dict(current.get("parameters", current.get("settings", current)))
        output: list[CandidatePatch] = []
        labels = {
            str(label)
            for item in experiences
            for label in item.get("failure_labels", [])
        }
        rules = list(current.get("rules", []))
        # The v0001 scalar novelty/query-similarity parameters are not
        # interpolated into rule conditions, so changing them alone is a no-op.
        # Generate an explicit allow-listed condition edit only when telemetry
        # proves that the unresolved signal was unavailable and duplicates were
        # observed.  The hard caps and action remain unchanged.
        if {"duplicate_search", "unresolved_signal_unavailable"} <= labels:
            revised = copy.deepcopy(rules)
            changed = False
            for rule in revised:
                if rule.get("id") != "rewrite_low_novelty":
                    continue
                conditions = rule.get("all", [])
                for index, condition in enumerate(conditions):
                    if condition.get("signal") == "unresolved_claims":
                        conditions[index] = {
                            "signal": "duplicate_ratio",
                            "operator": ">=",
                            "value": 0.4,
                        }
                        changed = True
                        break
            if changed:
                output.append(CandidatePatch(
                    "policy",
                    "rules",
                    rules,
                    revised,
                    f"{hypothesis} Replace unavailable unresolved-claim gating with observed duplicate evidence.",
                    patch_kind="rule_condition",
                ))
        hard_limits = dict(current.get("hard_limits", {}))
        if "search_hard_cap_reached" in labels:
            old_cap = hard_limits.get("max_search_attempts")
            if isinstance(old_cap, int) and old_cap < 100:
                output.append(CandidatePatch(
                    "policy",
                    "hard_limits.max_search_attempts",
                    old_cap,
                    min(old_cap + 2, 100),
                    f"{hypothesis} Diagnostic budget candidate; requires cost-gated replay.",
                    patch_kind="hard_limit",
                ))
        if "missed_query_rewrite" in labels:
            old_rewrites = settings.get("max_rewrites")
            if isinstance(old_rewrites, int) and old_rewrites < 10:
                output.append(CandidatePatch(
                    "policy",
                    "parameters.max_rewrites",
                    old_rewrites,
                    old_rewrites + 1,
                    hypothesis,
                    patch_kind="rewrite_budget",
                ))
        return output

    @staticmethod
    def _skill_candidates(
        current: Mapping[str, Any],
        hypothesis: str,
        experiences: Sequence[Mapping[str, Any]] = (),
        *,
        primitive_suggestions: Sequence[str] = (),
    ) -> list[CandidatePatch]:
        settings = dict(current.get("parameters", current.get("settings", {})))
        output: list[CandidatePatch] = []
        primitives = list(current.get("primitives", []))
        labels = {
            str(label)
            for item in experiences
            for label in item.get("failure_labels", [])
        }
        additions = [
            ("missing_entity_value_alignment", "align_entity_value"),
            ("missing_measurement_unit_conversion", "convert_measurement_unit"),
            ("missing_irrelevant_number_filter", "filter_semantically_irrelevant_numbers"),
        ]
        required_bundle_labels = {item[0] for item in additions}
        bundle_available = (
            required_bundle_labels <= labels
            and not set(CandidateGenerator.SAFE_PRIMITIVE_BUNDLE_MEMBERS) & set(primitives)
        )
        suggestion_rank = {
            name: index for index, name in enumerate(primitive_suggestions)
        }
        additions.sort(key=lambda item: suggestion_rank.get(item[1], len(suggestion_rank)))
        if bundle_available:
            revised = list(primitives)
            insertion_point = (
                revised.index("compare_value_unit_date")
                if "compare_value_unit_date" in revised
                else revised.index("classify_supported_contradicted_unknown")
            )
            revised[insertion_point:insertion_point] = CandidateGenerator.SAFE_PRIMITIVE_BUNDLE_MEMBERS
            output.append(CandidatePatch(
                "skill",
                "primitives",
                primitives,
                revised,
                (
                    f"{hypothesis} Atomic allow-listed profile {CandidateGenerator.SAFE_PRIMITIVE_BUNDLE} "
                    f"covers {', '.join(sorted(required_bundle_labels))}."
                ),
                patch_kind="atomic_bundle",
                bundle_id=CandidateGenerator.SAFE_PRIMITIVE_BUNDLE,
                added_primitives=CandidateGenerator.SAFE_PRIMITIVE_BUNDLE_MEMBERS,
            ))
        for failure_label, primitive in additions:
            if failure_label not in labels or primitive in primitives:
                continue
            revised = list(primitives)
            insertion_point = (
                revised.index("compare_value_unit_date")
                if "compare_value_unit_date" in revised
                else revised.index("classify_supported_contradicted_unknown")
            )
            revised.insert(insertion_point, primitive)
            output.append(CandidatePatch(
                "skill",
                "primitives",
                primitives,
                revised,
                f"{hypothesis} Trigger: {failure_label}; enable safe primitive {primitive}.",
                patch_kind="add_primitive",
                added_primitives=(primitive,),
            ))
        old_tolerance = settings.get("relative_tolerance")
        for new in (0.0, 0.005):
            if new != old_tolerance:
                output.append(CandidatePatch("skill", "parameters.relative_tolerance", old_tolerance, new, hypothesis))
        old_conflict = current.get("conflict_policy", "mark_unknown")
        for new in ("mark_unknown", "mark_contradicted"):
            if new != old_conflict:
                output.append(CandidatePatch("skill", "conflict_policy", old_conflict, new, hypothesis))
        priorities = list(current.get("source_priority", []))
        if len(priorities) >= 2:
            revised_priorities = list(priorities)
            revised_priorities[0], revised_priorities[1] = revised_priorities[1], revised_priorities[0]
            output.append(CandidatePatch("skill", "source_priority", priorities, revised_priorities, hypothesis))
        return output


def apply_candidate_patch(base: Mapping[str, Any], patch: CandidatePatch) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    keys = patch.field.split(".")
    cursor = result
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = patch.new
    return result
