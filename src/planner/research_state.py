"""A small, deterministic state graph for aspect-aware research.

The existing :mod:`planner.dag` describes *execution dependencies*.  This
module describes what is still unknown about a research question.  It is
deliberately independent from the orchestrator: a caller may use the graph
to choose a bounded action and then update it with whatever evidence the
caller obtained.

There are no model calls or random choices here.  Scores are made up of
named, inspectable components and all iteration/tie breaking is stable.  A
state can therefore be recorded and replayed with :meth:`to_json` and
:meth:`from_json`.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


__all__ = [
    "ActionScore",
    "Claim",
    "ClaimState",
    "DecisionRecord",
    "Facet",
    "FacetState",
    "FrontierAction",
    "OpenQuestion",
    "OpenQuestionState",
    "ResearchStateGraph",
]


class FrontierAction(str, Enum):
    """The only actions a frontier controller may emit.

    Values are intentionally lower-case and stable because they are part of
    replay artifacts and may be consumed by a downstream dispatcher.
    """

    SEARCH_NEW_FACET = "search_new_facet"
    DEEPEN_CLAIM = "deepen_claim"
    OPEN_PRIMARY_SOURCE = "open_primary_source"
    CROSS_VALIDATE = "cross_validate"
    RESOLVE_CONFLICT = "resolve_conflict"
    STOP = "stop"

    @classmethod
    def coerce(cls, value: "FrontierAction | str") -> "FrontierAction":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown frontier action {value!r}; expected one of {allowed}") from exc


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    if not math.isfinite(number):
        number = low
    return min(high, max(low, number))


def _normalise_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalise_ids(values: Iterable[Any] | None) -> list[str]:
    if values is None:
        return []
    # IDs are sorted for stable artifacts.  Keep one copy of each ID.
    return sorted({_normalise_id(item, "") for item in values if str(item or "").strip()})


def _text_tokens(text: str) -> set[str]:
    # This is only for the deterministic importance heuristic; it is not a
    # relevance model and intentionally has no locale-specific randomness.
    return {token for token in re.findall(r"[\w\u4e00-\u9fff]+", text.casefold()) if len(token) > 1}


@dataclass
class FacetState:
    """Coverage state for one research aspect/perspective."""

    facet_id: str
    description: str = ""
    importance: float = 0.5
    critical: bool = False
    expected_questions: list[str] = field(default_factory=list)
    coverage: float = 0.0
    support: float = 0.0
    source_diversity: float = 0.0
    conflict: float = 0.0
    freshness: float = 1.0
    cost_so_far: float = 0.0
    claim_ids: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    attempted_actions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.facet_id = _normalise_id(self.facet_id, "facet")
        self.importance = _clamp(self.importance)
        self.coverage = _clamp(self.coverage)
        self.support = _clamp(self.support)
        self.source_diversity = _clamp(self.source_diversity)
        self.conflict = _clamp(self.conflict)
        self.freshness = _clamp(self.freshness)
        self.cost_so_far = max(0.0, float(self.cost_so_far or 0.0))
        self.expected_questions = sorted({str(item) for item in self.expected_questions if str(item).strip()})
        self.claim_ids = _normalise_ids(self.claim_ids)
        self.dependencies = _normalise_ids(self.dependencies)
        self.attempted_actions = {
            str(key): max(0, int(value)) for key, value in (self.attempted_actions or {}).items()
        }

    @property
    def uncovered(self) -> float:
        return 1.0 - self.coverage


@dataclass
class ClaimState:
    """State for one checkable claim belonging to a facet."""

    claim_id: str
    facet_id: str
    text: str = ""
    importance: float = 0.5
    risk: str = "normal"
    support: float = 0.0
    status: str = "unknown"  # unknown | partial | supported | contradicted
    source_ids: list[str] = field(default_factory=list)
    primary_source_ids: list[str] = field(default_factory=list)
    conflict: float = 0.0
    cost_so_far: float = 0.0
    attempted_actions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.claim_id = _normalise_id(self.claim_id, "claim")
        self.facet_id = _normalise_id(self.facet_id, "facet")
        self.importance = _clamp(self.importance)
        self.support = _clamp(self.support)
        self.risk = str(self.risk or "normal").casefold()
        self.status = str(self.status or "unknown").casefold()
        if self.status not in {"unknown", "partial", "supported", "contradicted"}:
            self.status = "unknown"
        self.source_ids = _normalise_ids(self.source_ids)
        self.primary_source_ids = _normalise_ids(self.primary_source_ids)
        self.conflict = _clamp(self.conflict)
        self.cost_so_far = max(0.0, float(self.cost_so_far or 0.0))
        self.attempted_actions = {
            str(key): max(0, int(value)) for key, value in (self.attempted_actions or {}).items()
        }

    @property
    def high_risk(self) -> bool:
        return self.risk in {"high", "critical"} or self.importance >= 0.8

    @property
    def unresolved(self) -> bool:
        return self.status not in {"supported"} or self.support < 1.0


@dataclass
class OpenQuestionState:
    """A question for which the graph has not yet obtained an answer."""

    question_id: str
    text: str
    facet_id: str | None = None
    importance: float = 0.5
    critical: bool = False
    severity: float = 0.5
    resolved: bool = False
    conflict: float = 0.0

    def __post_init__(self) -> None:
        self.question_id = _normalise_id(self.question_id, "question")
        self.text = str(self.text or "")
        self.facet_id = _normalise_id(self.facet_id, "") if self.facet_id else None
        self.importance = _clamp(self.importance)
        self.severity = _clamp(self.severity)
        self.conflict = _clamp(self.conflict)
        self.resolved = bool(self.resolved)


# Friendly aliases used by callers that prefer the shorter names from the
# design document.  The canonical class remains OpenQuestionState so its
# name cannot be confused with orchestrator.schemas.ResearchReport fields.
OpenQuestion = OpenQuestionState
Facet = FacetState
Claim = ClaimState


@dataclass
class ActionScore:
    """Transparent score for one action/target pair."""

    action: FrontierAction
    target_id: str | None
    value: float
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "value": round(float(self.value), 10),
            "components": {key: round(float(self.components[key]), 10) for key in sorted(self.components)},
            "reason": self.reason,
        }


@dataclass
class DecisionRecord:
    """One selected/rejected frontier decision, suitable for replay."""

    action: FrontierAction
    target_id: str | None = None
    value: float = 0.0
    reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    budget_before: float = 0.0
    budget_after: float = 0.0
    consecutive_count: int = 0
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "value": round(float(self.value), 10),
            "reason": self.reason,
            "candidates": self.candidates,
            "budget_before": round(float(self.budget_before), 10),
            "budget_after": round(float(self.budget_after), 10),
            "consecutive_count": int(self.consecutive_count),
            "accepted": bool(self.accepted),
        }


class ResearchStateGraph:
    """Deterministic research frontier controller.

    ``budget_limit`` and action costs are abstract units.  A caller can map a
    unit to backend calls, tokens, or wall-clock cost, while the graph keeps a
    single hard cap.  State transitions are explicit: selecting an action does
    not pretend that evidence was found; call :meth:`record_action` and then
    update the relevant facet/claim/question with observed evidence.
    """

    ACTION_ORDER: tuple[FrontierAction, ...] = (
        FrontierAction.SEARCH_NEW_FACET,
        FrontierAction.DEEPEN_CLAIM,
        FrontierAction.OPEN_PRIMARY_SOURCE,
        FrontierAction.CROSS_VALIDATE,
        FrontierAction.RESOLVE_CONFLICT,
        FrontierAction.STOP,
    )
    DEFAULT_ACTION_COSTS: dict[FrontierAction, float] = {
        FrontierAction.SEARCH_NEW_FACET: 1.0,
        FrontierAction.DEEPEN_CLAIM: 1.0,
        FrontierAction.OPEN_PRIMARY_SOURCE: 1.0,
        FrontierAction.CROSS_VALIDATE: 1.0,
        FrontierAction.RESOLVE_CONFLICT: 1.0,
        FrontierAction.STOP: 0.0,
    }

    def __init__(
        self,
        query: str,
        facets: Iterable[FacetState] | Mapping[str, FacetState] | None = None,
        claims: Iterable[ClaimState] | Mapping[str, ClaimState] | None = None,
        open_questions: Iterable[OpenQuestionState] | Mapping[str, OpenQuestionState] | None = None,
        *,
        budget_limit: float = 8.0,
        budget_used: float = 0.0,
        max_consecutive_actions: int = 2,
        action_limits: Mapping[FrontierAction | str, int] | None = None,
        action_costs: Mapping[FrontierAction | str, float] | None = None,
        critical_coverage_threshold: float = 0.8,
        high_risk_support_threshold: float = 0.8,
        marginal_gain_threshold: float = 0.15,
        lambda_cost: float = 0.5,
        decision_trace: Iterable[DecisionRecord | Mapping[str, Any]] | None = None,
    ) -> None:
        self.query = str(query or "")
        self.facets: dict[str, FacetState] = self._coerce_collection(facets, FacetState)
        self.claims: dict[str, ClaimState] = self._coerce_collection(claims, ClaimState)
        self.open_questions: dict[str, OpenQuestionState] = self._coerce_collection(open_questions, OpenQuestionState)
        self.budget_limit = max(0.0, float(budget_limit))
        self.budget_used = max(0.0, float(budget_used))
        self.max_consecutive_actions = max(1, int(max_consecutive_actions))
        self.action_limits: dict[FrontierAction, int] = {}
        for action, limit in (action_limits or {}).items():
            self.action_limits[FrontierAction.coerce(action)] = max(0, int(limit))
        self.action_costs = dict(self.DEFAULT_ACTION_COSTS)
        for action, cost in (action_costs or {}).items():
            self.action_costs[FrontierAction.coerce(action)] = max(0.0, float(cost))
        self.critical_coverage_threshold = _clamp(critical_coverage_threshold)
        self.high_risk_support_threshold = _clamp(high_risk_support_threshold)
        self.marginal_gain_threshold = max(0.0, float(marginal_gain_threshold))
        self.lambda_cost = max(0.0, float(lambda_cost))
        self.action_counts: dict[FrontierAction, int] = {action: 0 for action in self.ACTION_ORDER}
        self.last_action: FrontierAction | None = None
        self.consecutive_action_count = 0
        self.decision_trace: list[DecisionRecord] = []
        for item in decision_trace or []:
            self.decision_trace.append(self._coerce_decision(item))
            action = self.decision_trace[-1].action
            if self.decision_trace[-1].accepted:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                if action == self.last_action:
                    self.consecutive_action_count += 1
                else:
                    self.last_action = action
                    self.consecutive_action_count = 1
        self._link_claims_to_facets()

    # ------------------------------------------------------------------
    # Construction and serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_collection(values: Any, cls: type) -> dict[str, Any]:
        if values is None:
            return {}
        if isinstance(values, Mapping):
            iterable = values.values()
        else:
            iterable = values
        result: dict[str, Any] = {}
        for value in iterable:
            item = value if isinstance(value, cls) else cls(**dict(value))
            key = item.facet_id if isinstance(item, FacetState) else item.claim_id if isinstance(item, ClaimState) else item.question_id
            result[str(key)] = item
        return {key: result[key] for key in sorted(result)}

    @classmethod
    def from_subtasks(
        cls,
        query: str,
        subtasks: Sequence[Any] | Iterable[Any],
        dag: Any | None = None,
        **kwargs: Any,
    ) -> "ResearchStateGraph":
        """Build facets from ``SubTask`` objects using a stable heuristic.

        By default one subtask becomes one facet (using its task ID as the
        stable ID), making the mapping easy to join with existing DAG
        execution results.  Tasks carrying the same optional ``facet_id``
        are merged into one aspect.
        Priority one tasks are critical; if every task has the same priority,
        all are critical.  Description/query token overlap adds a bounded
        importance bonus but never changes ordering for equal inputs.
        """
        def task_value(task: Any, name: str, default: Any = None) -> Any:
            if isinstance(task, Mapping):
                return task.get(name, default)
            return getattr(task, name, default)

        def values(value: Any) -> list[Any]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        ordered = sorted(
            list(subtasks),
            key=lambda task: (
                int(task_value(task, "priority", 1) or 1),
                str(task_value(task, "task_id", "")),
            ),
        )
        if not ordered:
            return cls(query, **kwargs)
        priorities = [int(task_value(task, "priority", 1) or 1) for task in ordered]
        minimum = min(priorities)
        query_tokens = _text_tokens(query)
        # ``facet_id`` is an optional extension to SubTask.  Several task
        # nodes may therefore represent one aspect; merge them before adding
        # the facet so the state graph remains a layer above the execution
        # DAG.  A task ID is used only when no facet ID is supplied.
        groups: dict[str, list[Any]] = {}
        for task in ordered:
            metadata = task_value(task, "metadata", {}) or {}
            facet_hint = task_value(task, "facet_id", None) or (
                metadata.get("facet_id") if isinstance(metadata, Mapping) else None
            )
            task_id = _normalise_id(task_value(task, "task_id", None), f"facet_{len(groups) + 1}")
            group_id = _normalise_id(facet_hint, task_id)
            groups.setdefault(group_id, []).append(task)
        task_to_facet = {
            _normalise_id(task_value(task, "task_id", None), f"task_{index}"): group_id
            for index, (group_id, group) in enumerate(groups.items(), start=1)
            for task in group
        }
        facets: list[FacetState] = []
        claims: list[ClaimState] = []
        open_questions: list[OpenQuestionState] = []
        for facet_id in sorted(groups):
            group = groups[facet_id]
            descriptions = [
                str(task_value(task, "description", "") or task_value(task, "task_id", facet_id))
                for task in group
            ]
            descriptions = list(dict.fromkeys(descriptions))
            description = descriptions[0] if descriptions else facet_id
            priorities_for_group = [int(task_value(task, "priority", 1) or 1) for task in group]
            priority = min(priorities_for_group)
            overlap = len(query_tokens & _text_tokens(" ".join(descriptions))) / max(1, len(query_tokens))
            importance = _clamp(0.55 + 0.25 * (1.0 / max(1, priority)) + 0.20 * overlap)
            explicit_critical = any(bool(task_value(task, "critical", False)) for task in group)
            dependencies: set[str] = set()
            expected: list[str] = []
            for task in group:
                task_id = _normalise_id(task_value(task, "task_id", None), facet_id)
                dependencies.update(str(dep) for dep in values(task_value(task, "dependencies", [])))
                if dag is not None and hasattr(dag, "get_dependencies"):
                    dependencies.update(str(dep) for dep in (dag.get_dependencies(task_id) or []))
                custom_questions = [
                    *values(task_value(task, "expected_questions", [])),
                    *values(task_value(task, "completion_criteria", [])),
                ]
                expected.extend(str(question) for question in custom_questions if str(question).strip())
                for claim_id in values(task_value(task, "claim_ids", [])):
                    claims.append(ClaimState(
                        claim_id=str(claim_id),
                        facet_id=facet_id,
                        text=str(task_value(task, "description", "") or ""),
                        importance=importance,
                        risk="high" if explicit_critical or priority == minimum else "normal",
                    ))
                risk_question = str(task_value(task, "risk_question", "") or "").strip()
                if risk_question:
                    open_questions.append(OpenQuestionState(
                        question_id=f"risk_{facet_id}_{len(open_questions) + 1}",
                        text=risk_question,
                        facet_id=facet_id,
                        importance=importance,
                        critical=explicit_critical or priority == minimum,
                        severity=importance,
                    ))
            expected.extend(descriptions)
            mapped_dependencies = {task_to_facet.get(dep, dep) for dep in dependencies} - {facet_id}
            facets.append(
                FacetState(
                    facet_id=facet_id,
                    description=description,
                    importance=importance,
                    critical=explicit_critical or priority == minimum,
                    expected_questions=expected,
                    dependencies=sorted(mapped_dependencies),
                )
            )
        return cls(
            query,
            facets=facets,
            claims=claims,
            open_questions=open_questions,
            **kwargs,
        )

    @classmethod
    def from_dag(cls, query: str, dag: Any, subtasks: Sequence[Any] | Iterable[Any], **kwargs: Any) -> "ResearchStateGraph":
        """Alias with the DAG-first argument order used by some callers."""
        return cls.from_subtasks(query, subtasks, dag=dag, **kwargs)

    @classmethod
    def from_plan(cls, query: str, subtasks: Sequence[Any] | Iterable[Any], dag: Any | None = None, **kwargs: Any) -> "ResearchStateGraph":
        return cls.from_subtasks(query, subtasks, dag=dag, **kwargs)

    def _link_claims_to_facets(self) -> None:
        for claim in self.claims.values():
            facet = self.facets.get(claim.facet_id)
            if facet is not None and claim.claim_id not in facet.claim_ids:
                facet.claim_ids.append(claim.claim_id)
                facet.claim_ids.sort()

    @staticmethod
    def _coerce_decision(item: DecisionRecord | Mapping[str, Any]) -> DecisionRecord:
        if isinstance(item, DecisionRecord):
            return item
        payload = dict(item)
        payload["action"] = FrontierAction.coerce(payload.get("action", FrontierAction.STOP))
        return DecisionRecord(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "facets": [self._serialise_dataclass(self.facets[key]) for key in sorted(self.facets)],
            "claims": [self._serialise_dataclass(self.claims[key]) for key in sorted(self.claims)],
            "open_questions": [self._serialise_dataclass(self.open_questions[key]) for key in sorted(self.open_questions)],
            "budget_limit": self.budget_limit,
            "budget_used": self.budget_used,
            "remaining_budget": self.remaining_budget,
            "max_consecutive_actions": self.max_consecutive_actions,
            "action_limits": {action.value: self.action_limits[action] for action in sorted(self.action_limits, key=lambda x: x.value)},
            "action_costs": {action.value: self.action_costs[action] for action in sorted(self.action_costs, key=lambda x: x.value)},
            "critical_coverage_threshold": self.critical_coverage_threshold,
            "high_risk_support_threshold": self.high_risk_support_threshold,
            "marginal_gain_threshold": self.marginal_gain_threshold,
            "lambda_cost": self.lambda_cost,
            "action_counts": {action.value: self.action_counts.get(action, 0) for action in self.ACTION_ORDER},
            "last_action": self.last_action.value if self.last_action else None,
            "consecutive_action_count": self.consecutive_action_count,
            "decision_trace": [record.to_dict() for record in self.decision_trace],
        }

    @staticmethod
    def _serialise_dataclass(value: Any) -> dict[str, Any]:
        return asdict(value)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchStateGraph":
        data = dict(payload)
        graph = cls(
            data.get("query", ""),
            facets=data.get("facets", []),
            claims=data.get("claims", []),
            open_questions=data.get("open_questions", []),
            budget_limit=data.get("budget_limit", 8.0),
            budget_used=data.get("budget_used", 0.0),
            max_consecutive_actions=data.get("max_consecutive_actions", 2),
            action_limits=data.get("action_limits", {}),
            action_costs=data.get("action_costs", {}),
            critical_coverage_threshold=data.get("critical_coverage_threshold", 0.8),
            high_risk_support_threshold=data.get("high_risk_support_threshold", 0.8),
            marginal_gain_threshold=data.get("marginal_gain_threshold", 0.15),
            lambda_cost=data.get("lambda_cost", 0.5),
            decision_trace=data.get("decision_trace", []),
        )
        # Keep explicit counters when loading a trace produced by an older
        # implementation, while never trusting arbitrary non-enum keys.
        for key, value in (data.get("action_counts", {}) or {}).items():
            try:
                graph.action_counts[FrontierAction.coerce(key)] = max(0, int(value))
            except ValueError:
                continue
        last = data.get("last_action")
        graph.last_action = FrontierAction.coerce(last) if last else graph.last_action
        graph.consecutive_action_count = max(0, int(data.get("consecutive_action_count", graph.consecutive_action_count)))
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "ResearchStateGraph":
        return cls.from_dict(json.loads(payload))

    # ------------------------------------------------------------------
    # State mutation helpers
    # ------------------------------------------------------------------

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget_limit - self.budget_used)

    @property
    def budget_exhausted(self) -> bool:
        return self.remaining_budget <= 1e-12

    def add_facet(self, facet: FacetState | Mapping[str, Any]) -> FacetState:
        item = facet if isinstance(facet, FacetState) else FacetState(**dict(facet))
        self.facets[item.facet_id] = item
        self.facets = {key: self.facets[key] for key in sorted(self.facets)}
        return item

    def add_claim(self, claim: ClaimState | Mapping[str, Any]) -> ClaimState:
        item = claim if isinstance(claim, ClaimState) else ClaimState(**dict(claim))
        self.claims[item.claim_id] = item
        self.claims = {key: self.claims[key] for key in sorted(self.claims)}
        self._link_claims_to_facets()
        return item

    def add_open_question(self, question: OpenQuestionState | Mapping[str, Any]) -> OpenQuestionState:
        item = question if isinstance(question, OpenQuestionState) else OpenQuestionState(**dict(question))
        self.open_questions[item.question_id] = item
        self.open_questions = {key: self.open_questions[key] for key in sorted(self.open_questions)}
        return item

    def update_facet(self, facet_id: str, **updates: Any) -> FacetState:
        facet = self.facets[str(facet_id)]
        for key, value in updates.items():
            if not hasattr(facet, key):
                raise AttributeError(f"unknown facet field: {key}")
            setattr(facet, key, value)
        facet.__post_init__()
        return facet

    def update_claim(self, claim_id: str, **updates: Any) -> ClaimState:
        claim = self.claims[str(claim_id)]
        for key, value in updates.items():
            if not hasattr(claim, key):
                raise AttributeError(f"unknown claim field: {key}")
            setattr(claim, key, value)
        claim.__post_init__()
        self._link_claims_to_facets()
        return claim

    def resolve_question(self, question_id: str) -> OpenQuestionState:
        question = self.open_questions[str(question_id)]
        question.resolved = True
        return question

    # ------------------------------------------------------------------
    # Deterministic frontier scoring
    # ------------------------------------------------------------------

    def _target_for(self, action: FrontierAction) -> str | None:
        candidates = self._candidate_scores(action)
        if not candidates:
            return None
        return candidates[0].target_id

    def _candidate_scores(self, action: FrontierAction) -> list[ActionScore]:
        action = FrontierAction.coerce(action)
        if action is FrontierAction.STOP:
            return [ActionScore(action, None, 0.0, {"baseline": 0.0}, "explicit stop")]
        if action is FrontierAction.SEARCH_NEW_FACET:
            return [self._score_facet_action(action, facet) for facet in self.facets.values() if facet.coverage < 1.0]
        if action is FrontierAction.DEEPEN_CLAIM:
            return [self._score_claim_action(action, claim) for claim in self.claims.values() if claim.unresolved]
        if action is FrontierAction.OPEN_PRIMARY_SOURCE:
            return [self._score_claim_action(action, claim) for claim in self.claims.values() if not claim.primary_source_ids]
        if action is FrontierAction.CROSS_VALIDATE:
            return [self._score_claim_action(action, claim) for claim in self.claims.values() if claim.source_ids and len(claim.source_ids) < 2]
        if action is FrontierAction.RESOLVE_CONFLICT:
            claim_scores = [self._score_claim_action(action, claim) for claim in self.claims.values() if claim.conflict > 0]
            facet_scores = [self._score_facet_action(action, facet) for facet in self.facets.values() if facet.conflict > 0]
            return claim_scores + facet_scores
        return []

    def _normalised_cost(self, action: FrontierAction) -> float:
        cost = self.action_costs.get(action, 1.0)
        if self.budget_limit <= 0:
            return 1.0 if cost else 0.0
        return _clamp(cost / self.budget_limit)

    def _duplicate_risk(self, action: FrontierAction, target: Any) -> float:
        attempts = int(getattr(target, "attempted_actions", {}).get(action.value, 0))
        # Repeated attempts asymptotically approach one; this makes the
        # penalty visible while preserving a deterministic, bounded score.
        return _clamp(attempts / 2.0)

    def _score_facet_action(self, action: FrontierAction, facet: FacetState) -> ActionScore:
        gap = facet.uncovered
        importance = 0.75 + 0.25 * facet.importance if facet.critical else facet.importance
        diversity_gap = 1.0 - facet.source_diversity
        conflict_reduction = facet.conflict if action is FrontierAction.RESOLVE_CONFLICT else 0.0
        coverage_gain = gap * importance
        support_gain = gap * facet.importance * 0.25 if action is FrontierAction.SEARCH_NEW_FACET else 0.0
        diversity_gain = diversity_gap * importance * (0.5 if action is FrontierAction.SEARCH_NEW_FACET else 0.0)
        duplicate_risk = self._duplicate_risk(action, facet)
        components = {
            "expected_coverage_gain": coverage_gain,
            "expected_support_gain": support_gain,
            "expected_source_diversity_gain": diversity_gain,
            "conflict_reduction": conflict_reduction,
            "duplicate_risk": duplicate_risk,
            "normalized_cost": self._normalised_cost(action),
        }
        value = (
            coverage_gain
            + 1.2 * support_gain
            + 0.6 * diversity_gain
            + 0.8 * conflict_reduction
            - 0.5 * duplicate_risk
            - self.lambda_cost * components["normalized_cost"]
        )
        reason = "critical facet has uncovered questions" if facet.critical else "facet has uncovered questions"
        return ActionScore(action, facet.facet_id, value, components, reason)

    def _score_claim_action(self, action: FrontierAction, claim: ClaimState) -> ActionScore:
        facet = self.facets.get(claim.facet_id)
        importance = 0.75 + 0.25 * claim.importance if claim.high_risk else claim.importance
        if facet is not None and facet.critical:
            importance = min(1.0, importance + 0.15)
        support_gap = 1.0 - claim.support
        coverage_gain = support_gap * importance * (0.6 if action is FrontierAction.DEEPEN_CLAIM else 0.2)
        support_gain = support_gap * importance
        primary_gap = 1.0 if not claim.primary_source_ids else 0.0
        diversity_gap = _clamp(1.0 - min(1.0, len(claim.source_ids) / 2.0))
        diversity_gain = diversity_gap
        if action is FrontierAction.OPEN_PRIMARY_SOURCE:
            coverage_gain *= 0.6
            diversity_gain = max(diversity_gap, 0.5)
        elif action is FrontierAction.CROSS_VALIDATE:
            coverage_gain *= 0.35
            support_gain *= 0.45
        elif action is FrontierAction.RESOLVE_CONFLICT:
            coverage_gain = 0.0
            support_gain = 0.0
            diversity_gain = 0.0
        conflict_reduction = claim.conflict if action is FrontierAction.RESOLVE_CONFLICT else 0.0
        duplicate_risk = self._duplicate_risk(action, claim)
        components = {
            "expected_coverage_gain": coverage_gain,
            "expected_support_gain": support_gain,
            "expected_source_diversity_gain": diversity_gain * 0.6,
            "conflict_reduction": conflict_reduction,
            "duplicate_risk": duplicate_risk,
            "normalized_cost": self._normalised_cost(action),
            "primary_source_gap": primary_gap,
        }
        value = (
            coverage_gain
            + 1.2 * support_gain
            + 0.6 * components["expected_source_diversity_gain"]
            + 0.8 * conflict_reduction
            - 0.5 * duplicate_risk
            - self.lambda_cost * components["normalized_cost"]
        )
        if action is FrontierAction.OPEN_PRIMARY_SOURCE and primary_gap:
            value += 0.25 * importance
        if action is FrontierAction.CROSS_VALIDATE and len(claim.source_ids) < 2:
            value += 0.15 * importance
        reason = {
            FrontierAction.DEEPEN_CLAIM: "claim lacks sufficient support",
            FrontierAction.OPEN_PRIMARY_SOURCE: "high-risk claim lacks a primary source",
            FrontierAction.CROSS_VALIDATE: "claim has too few independent sources",
            FrontierAction.RESOLVE_CONFLICT: "claim has unresolved conflicting evidence",
        }.get(action, "claim frontier")
        return ActionScore(action, claim.claim_id, value, components, reason)

    def score_action(self, action: FrontierAction | str, target_id: str | None = None) -> float:
        """Return the value of an action, or zero when no target exists."""
        action = FrontierAction.coerce(action)
        candidates = self._candidate_scores(action)
        if target_id is not None:
            candidates = [item for item in candidates if item.target_id == str(target_id)]
        if not candidates:
            return 0.0
        return max(item.value for item in candidates)

    def score_details(self, action: FrontierAction | str, target_id: str | None = None) -> ActionScore | None:
        action = FrontierAction.coerce(action)
        candidates = self._candidate_scores(action)
        if target_id is not None:
            candidates = [item for item in candidates if item.target_id == str(target_id)]
        return self._best_score(candidates)

    @staticmethod
    def _best_score(candidates: Sequence[ActionScore]) -> ActionScore | None:
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item.value, item.target_id or ""))[0]

    def rank_frontier(self, include_stop: bool = True) -> list[ActionScore]:
        """Return best candidate for each currently possible action.

        Sorting uses value first, then the fixed enum order, then target ID;
        no dictionary/set iteration can affect the result.
        """
        ranked: list[ActionScore] = []
        for order, action in enumerate(self.ACTION_ORDER):
            if action is FrontierAction.STOP and not include_stop:
                continue
            best = self._best_score(self._candidate_scores(action))
            if best is not None:
                ranked.append(best)
        order_map = {action: index for index, action in enumerate(self.ACTION_ORDER)}
        return sorted(ranked, key=lambda item: (-item.value, order_map[item.action], item.target_id or ""))

    def frontier_scores(self, include_stop: bool = True) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.rank_frontier(include_stop=include_stop)]

    # ------------------------------------------------------------------
    # Budgeted decisions and stop gate
    # ------------------------------------------------------------------

    def _limit_for(self, action: FrontierAction) -> int:
        return self.action_limits.get(action, self.max_consecutive_actions)

    def can_take_action(self, action: FrontierAction | str) -> bool:
        action = FrontierAction.coerce(action)
        if action is FrontierAction.STOP:
            return True
        if self.budget_exhausted or self.remaining_budget + 1e-12 < self.action_costs.get(action, 1.0):
            return False
        if action == self.last_action and self.consecutive_action_count >= self._limit_for(action):
            return False
        return True

    def choose_action(self, *, commit: bool = False) -> ActionScore:
        """Choose the highest-value permitted action.

        ``commit=False`` (the default) is a pure read.  ``commit=True`` calls
        :meth:`record_action`, useful for small synchronous controllers.
        """
        gate = self.stop_gate()
        if gate["should_stop"]:
            selected = ActionScore(FrontierAction.STOP, None, 0.0, {"baseline": 0.0}, gate["reason"])
        else:
            selected = next(
                (candidate for candidate in self.rank_frontier(include_stop=False) if self.can_take_action(candidate.action)),
                ActionScore(FrontierAction.STOP, None, 0.0, {"baseline": 0.0}, "no permitted frontier action"),
            )
        if commit:
            self.record_action(selected.action, target_id=selected.target_id, value=selected.value, reason=selected.reason)
        return selected

    def next_action(self, *, commit: bool = False) -> ActionScore:
        """Alias for :meth:`choose_action`."""
        return self.choose_action(commit=commit)

    def step(self) -> ActionScore:
        """Choose and commit one bounded action."""
        return self.choose_action(commit=True)

    def record_action(
        self,
        action: FrontierAction | str,
        target_id: str | None = None,
        *,
        cost: float | None = None,
        value: float | None = None,
        reason: str = "",
    ) -> bool:
        """Reserve budget and append an auditable decision.

        Rejected actions are also traced, but do not consume budget.  This is
        important when a replay explains why a controller stopped at a hard
        cap or continuity limit.
        """
        action = FrontierAction.coerce(action)
        before = self.remaining_budget
        actual_cost = self.action_costs.get(action, 1.0) if cost is None else max(0.0, float(cost))
        accepted = action is FrontierAction.STOP or self.can_take_action(action)
        if accepted and action is not FrontierAction.STOP:
            if before + 1e-12 < actual_cost:
                accepted = False
            else:
                self.budget_used += actual_cost
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                if action == self.last_action:
                    self.consecutive_action_count += 1
                else:
                    self.last_action = action
                    self.consecutive_action_count = 1
                target = self._target_object(action, target_id)
                if target is not None:
                    attempts = getattr(target, "attempted_actions")
                    attempts[action.value] = attempts.get(action.value, 0) + 1
                    if hasattr(target, "cost_so_far"):
                        target.cost_so_far += actual_cost
        if not accepted and not reason:
            reason = "budget_exhausted" if self.budget_exhausted else "action_limit_reached"
        if action is FrontierAction.STOP and not reason:
            reason = self.stop_gate()["reason"]
        self.decision_trace.append(
            DecisionRecord(
                action=action,
                target_id=target_id,
                value=self.score_action(action, target_id) if value is None else float(value),
                reason=reason,
                candidates=self.frontier_scores(),
                budget_before=before,
                budget_after=self.remaining_budget,
                consecutive_count=self.consecutive_action_count,
                accepted=accepted,
            )
        )
        return accepted

    def _target_object(self, action: FrontierAction, target_id: str | None) -> Any | None:
        if target_id is None:
            target_id = self._target_for(action)
        if target_id is None:
            return None
        if action is FrontierAction.SEARCH_NEW_FACET or action is FrontierAction.RESOLVE_CONFLICT and target_id in self.facets:
            return self.facets.get(target_id)
        return self.claims.get(target_id) or self.facets.get(target_id)

    def _critical_facets_covered(self) -> bool:
        critical = [facet for facet in self.facets.values() if facet.critical]
        return all(facet.coverage >= self.critical_coverage_threshold for facet in critical)

    def _high_risk_claims_supported(self) -> bool:
        high_risk = [claim for claim in self.claims.values() if claim.high_risk]
        return all(
            claim.support >= self.high_risk_support_threshold and claim.status != "contradicted"
            for claim in high_risk
        )

    def _unresolved_critical_conflicts(self) -> list[str]:
        conflicts: list[str] = []
        for facet in self.facets.values():
            if facet.critical and facet.conflict > 0:
                conflicts.append(f"facet:{facet.facet_id}")
        for claim in self.claims.values():
            facet = self.facets.get(claim.facet_id)
            if (claim.conflict > 0 or claim.status == "contradicted") and (
                claim.high_risk or facet is not None and facet.critical
            ):
                conflicts.append(f"claim:{claim.claim_id}")
        for question in self.open_questions.values():
            if question.critical and not question.resolved and question.conflict > 0:
                conflicts.append(f"question:{question.question_id}")
        return sorted(set(conflicts))

    def stop_gate(self) -> dict[str, Any]:
        """Evaluate the multi-condition stop gate.

        Budget exhaustion always stops execution, but ``complete`` remains
        false when a coverage/support/conflict condition is unmet.  Callers
        can therefore honestly report a partial result instead of treating a
        hard cap as successful research.
        """
        critical_covered = self._critical_facets_covered()
        high_risk_supported = self._high_risk_claims_supported()
        conflicts = self._unresolved_critical_conflicts()
        frontier = self.rank_frontier(include_stop=False)
        permitted_values = [item.value for item in frontier if self.can_take_action(item.action)]
        max_value = max(permitted_values, default=0.0)
        marginal_done = max_value <= self.marginal_gain_threshold
        complete = critical_covered and high_risk_supported and not conflicts
        if self.budget_exhausted:
            reason = "budget_exhausted"
            should_stop = True
        elif not critical_covered:
            reason = "critical_facets_uncovered"
            should_stop = False
        elif not high_risk_supported:
            reason = "high_risk_claims_unsupported"
            should_stop = False
        elif conflicts:
            reason = "critical_conflicts_unresolved"
            should_stop = False
        elif not marginal_done:
            reason = "frontier_value_above_threshold"
            should_stop = False
        else:
            reason = "stop_gate_satisfied"
            should_stop = True
        unresolved_questions = [
            {
                "question_id": question.question_id,
                "facet_id": question.facet_id,
                "text": question.text,
                "critical": question.critical,
                "severity": question.severity,
            }
            for question in self.open_questions.values()
            if not question.resolved
        ]
        return {
            "should_stop": should_stop,
            "stop": should_stop,
            "complete": complete,
            "honest": True,
            "reason": reason,
            "stop_reason": reason,
            "critical_facets_covered": critical_covered,
            "high_risk_claim_support": high_risk_supported,
            "unresolved_critical_conflicts": conflicts,
            "max_frontier_action_value": round(max_value, 10),
            "marginal_gain_below_threshold": marginal_done,
            "budget_exhausted": self.budget_exhausted,
            "remaining_budget": self.remaining_budget,
            "coverage_map": {
                facet.facet_id: round(facet.coverage, 10) for facet in self.facets.values()
            },
            "unresolved_questions": unresolved_questions,
        }

    def should_stop(self) -> bool:
        return bool(self.stop_gate()["should_stop"])
