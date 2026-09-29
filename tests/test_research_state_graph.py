from __future__ import annotations

from src.orchestrator.schemas import SubTask, TaskType
from src.planner.dag import DAG
from src.planner.planner import Planner
from src.planner.research_state import (
    ClaimState,
    FacetState,
    FrontierAction,
    OpenQuestionState,
    ResearchStateGraph,
)


def test_value_scoring_prefers_the_relevant_frontier_and_exposes_components():
    graph = ResearchStateGraph(
        "battery safety",
        facets=[
            FacetState("background", "history", importance=0.3, coverage=0.0),
            FacetState("safety", "battery safety", importance=1.0, critical=True, coverage=0.0),
        ],
        budget_limit=10,
    )

    ranked = graph.rank_frontier(include_stop=False)
    assert ranked[0].action is FrontierAction.SEARCH_NEW_FACET
    assert ranked[0].target_id == "safety"
    assert ranked[0].components["expected_coverage_gain"] > ranked[0].components["normalized_cost"]
    assert graph.score_action(FrontierAction.SEARCH_NEW_FACET, "safety") == ranked[0].value


def test_stop_gate_requires_critical_facet_and_high_risk_support():
    graph = ResearchStateGraph(
        "q",
        facets=[FacetState("critical", "must answer", critical=True, coverage=1.0)],
        claims=[ClaimState("c1", "critical", "risky fact", risk="high", support=0.0)],
        budget_limit=10,
    )
    gate = graph.stop_gate()
    assert gate["should_stop"] is False
    assert gate["reason"] == "high_risk_claims_unsupported"

    graph.update_claim(
        "c1",
        support=1.0,
        status="supported",
        source_ids=["s1", "s2"],
        primary_source_ids=["s1"],
    )
    gate = graph.stop_gate()
    assert gate["should_stop"] is True
    assert gate["complete"] is True
    assert gate["honest"] is True


def test_budget_exhaustion_stops_honestly_without_claiming_completion():
    graph = ResearchStateGraph(
        "q",
        facets=[FacetState("critical", "must answer", critical=True)],
        budget_limit=1,
    )
    assert graph.record_action(FrontierAction.SEARCH_NEW_FACET, "critical") is True
    assert graph.remaining_budget == 0
    gate = graph.stop_gate()
    assert gate["should_stop"] is True
    assert gate["reason"] == "budget_exhausted"
    assert gate["complete"] is False
    assert gate["honest"] is True
    assert graph.record_action(FrontierAction.SEARCH_NEW_FACET, "critical") is False


def test_serialization_and_same_state_selection_are_deterministic():
    graph = ResearchStateGraph(
        "q",
        facets=[FacetState("b", "B", importance=0.5), FacetState("a", "A", importance=0.5)],
        claims=[ClaimState("c", "a", "claim", support=0.3)],
        open_questions=[OpenQuestionState("q1", "what remains?", facet_id="a", critical=True)],
        budget_limit=4,
    )
    first = graph.choose_action()
    second = ResearchStateGraph.from_json(graph.to_json()).choose_action()
    assert first.to_dict() == second.to_dict()
    assert graph.to_json() == ResearchStateGraph.from_json(graph.to_json()).to_json()


def test_from_subtasks_merges_facet_ids_and_maps_dag_dependencies():
    first = SubTask(
        "search_a",
        TaskType.SEARCH,
        "find primary evidence",
        priority=1,
        facet_id="evidence",
        claim_ids=["claim_a"],
        completion_criteria=["locate one primary source"],
        risk_question="Could the primary source contradict the summary?",
    )
    second = SubTask("open_a", TaskType.ANALYZE, "inspect the evidence", priority=2, dependencies=["search_a"])
    third = SubTask("other", TaskType.SEARCH, "independent context", priority=3, dependencies=["open_a"])
    second.facet_id = "evidence"
    dag = DAG()
    dag.add_edge("search_a", "open_a")
    dag.add_edge("open_a", "other")

    graph = ResearchStateGraph.from_subtasks("primary evidence", [third, second, first], dag=dag)
    assert list(graph.facets) == ["evidence", "other"]
    assert graph.facets["evidence"].dependencies == []
    assert graph.facets["evidence"].expected_questions == [
        "find primary evidence", "inspect the evidence", "locate one primary source"
    ]
    assert graph.facets["other"].dependencies == ["evidence"]
    assert graph.claims["claim_a"].facet_id == "evidence"
    assert next(iter(graph.open_questions.values())).facet_id == "evidence"


def test_from_subtasks_treats_scalar_criteria_as_one_item_not_characters():
    graph = ResearchStateGraph.from_subtasks("q", [{
        "task_id": "search",
        "description": "find evidence",
        "completion_criteria": "one primary source",
        "claim_ids": "claim_one",
    }])
    assert graph.facets["search"].expected_questions == [
        "find evidence", "one primary source"
    ]
    assert list(graph.claims) == ["claim_one"]


def test_planner_normalizes_scalar_list_fields_from_model_json():
    task = Planner(policy=lambda _: {})._deserialize_subtask({
        "task_id": "search",
        "dependencies": "prior_task",
        "search_hints": "official report",
        "claim_ids": "claim_one",
        "completion_criteria": "one primary source",
    })
    assert task.dependencies == ["prior_task"]
    assert task.search_hints == ["official report"]
    assert task.claim_ids == ["claim_one"]
    assert task.completion_criteria == ["one primary source"]
