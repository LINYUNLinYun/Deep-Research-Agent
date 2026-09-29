"""Versioned search policy runtime and SearchController integration tests."""
from __future__ import annotations

import asyncio

import pytest

from src.harness_evolution.policy import PolicyValidationError, SearchControlPolicy
from src.harness_evolution.experiment import _mark_policy_mechanism_changes
from src.core.runner import _load_search_control_policy
from src.tools.search_controller import SearchController


def _spec(action: str = "rewrite_uncovered_facets") -> dict:
    return {
        "artifact_id": "search_control_policy",
        "version": "v0002",
        "parent": "v0001",
        "decision_point": "after_search",
        "parameters": {
            "novelty_threshold": 0.3,
            "query_similarity_threshold": 0.75,
            "max_rewrites": 2,
        },
        "rules": [{
            "id": "low_novelty",
            "priority": 100,
            "all": [
                {"signal": "evidence_novelty", "operator": "<", "value": 0.5},
                {"signal": "remaining_search_budget", "operator": ">=", "value": 1},
            ],
            "action": {"type": action},
            "limits": {"max_activations_per_run": 2},
        }],
        "fallback": {"action": {"type": "accept_results"}},
        "hard_limits": {"max_search_attempts": 10, "max_total_rewrites": 2},
    }


class _Search:
    name = "web_search"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    async def execute(self, query: str, top_n: int = 5):
        self.calls.append(query)
        results = self.responses.pop(0)
        return {"query": query, "results": results, "total": len(results)}


def test_policy_rejects_unknown_schema_elements():
    spec = _spec()
    spec["rules"][0]["all"][0]["signal"] = "secret_signal"
    with pytest.raises(PolicyValidationError, match="unknown policy signal"):
        SearchControlPolicy(spec)

    spec = _spec()
    spec["rules"][0]["action"] = {"type": "run_python"}
    with pytest.raises(PolicyValidationError, match="unknown policy action"):
        SearchControlPolicy(spec)


def test_policy_overrides_evolvable_controller_thresholds():
    policy = SearchControlPolicy(_spec(), sha256="abc123")
    controller = SearchController(
        novelty_threshold=0.01,
        query_similarity_threshold=0.01,
        max_rewrites=0,
        policy=policy,
    )
    assert controller.novelty_threshold == 0.3
    assert controller.query_similarity_threshold == 0.75
    assert controller.max_rewrites == 2


def test_after_search_policy_rewrites_at_most_once_and_emits_version_metadata():
    policy = SearchControlPolicy(_spec(), sha256="abc123")
    controller = SearchController(policy=policy)
    tool = _Search([
        [],
        [{"title": "Official", "url": "https://example.com/report", "snippet": "42"}],
        # A recursive implementation would incorrectly consume this response.
        [{"title": "Unexpected", "url": "https://example.com/third", "snippet": "x"}],
    ])

    result = asyncio.run(controller.execute(
        tool,
        {"query": "company revenue"},
        context={"unresolved_claims": ["revenue"], "remaining_search_budget": 2},
    ))

    assert len(tool.calls) == 2
    assert result["results"][0]["title"] == "Official"
    assert result["_search_policy"]["action"] == "rewrite_uncovered_facets"
    assert result["_search_policy"]["artifact"] == {
        "id": "search_control_policy", "version": "v0002", "sha256": "abc123"
    }
    snapshot = controller.snapshot()
    assert snapshot["policy"] == result["_search_policy"]["artifact"]
    assert len(snapshot["policy_decisions"]) == 1


def test_no_policy_keeps_legacy_empty_result_rewrite():
    controller = SearchController()
    tool = _Search([
        [],
        [{"title": "Legacy", "url": "https://example.com/legacy", "snippet": "ok"}],
    ])
    result = asyncio.run(controller.execute(tool, {"query": "rare query"}))
    assert len(tool.calls) == 2
    assert result["results"][0]["title"] == "Legacy"
    assert "_search_policy" not in result


def test_rule_activation_limit_falls_back():
    spec = _spec(action="stop_search")
    spec["rules"][0]["limits"]["max_activations_per_run"] = 1
    policy = SearchControlPolicy(spec)
    signals = {
        "evidence_novelty": 0.0,
        "remaining_search_budget": 1,
    }
    assert policy.decide(signals).action == "stop_search"
    assert policy.decide(signals).action == "accept_results"


def test_runner_policy_loading_is_opt_in_and_defaults_to_production():
    assert _load_search_control_policy({}) is None
    policy = _load_search_control_policy({"harness_evolution": {"enabled": True}})
    assert policy.artifact_id == "search_control_policy"
    assert policy.version == "v0001"
    assert len(policy.sha256) == 64


def test_task_hard_cap_blocks_provider_before_next_call():
    spec = _spec()
    spec["hard_limits"]["max_search_attempts"] = 2
    controller = SearchController(policy=SearchControlPolicy(spec, sha256="cap"))
    tool = _Search([
        [{"title": "A", "url": "https://example.com/a", "snippet": "a"}],
        [{"title": "B", "url": "https://example.com/b", "snippet": "b"}],
        [{"title": "must-not-run", "url": "https://example.com/c", "snippet": "c"}],
    ])

    async def run():
        await controller.execute(tool, {"query": "first"})
        await controller.execute(tool, {"query": "second"})
        return await controller.execute(tool, {"query": "third"})

    blocked = asyncio.run(run())
    assert tool.calls == ["first", "second"]
    assert blocked["hard_cap_reached"] is True
    assert blocked["stop_search_requested"] is True
    assert blocked["_search_policy"]["rule_id"] == "hard_limit:max_search_attempts"
    assert controller.snapshot()["stats"] == {
        "calls": 3,
        "backend_calls": 2,
        "cache_hits": 0,
        "rewritten_queries": 0,
        "duplicate_results": 0,
        "new_results": 2,
        "blocked_calls": 1,
    }


def test_task_hard_cap_does_not_starve_other_workers():
    spec = _spec()
    spec["hard_limits"]["max_search_attempts"] = 2
    controller = SearchController(
        max_backend_calls=10,
        policy=SearchControlPolicy(spec, sha256="per-task-cap"),
    )
    tool = _Search([
        [{"title": "A1", "url": "https://example.com/a1", "snippet": "a1"}],
        [{"title": "A2", "url": "https://example.com/a2", "snippet": "a2"}],
        [{"title": "B1", "url": "https://example.com/b1", "snippet": "b1"}],
        [{"title": "B2", "url": "https://example.com/b2", "snippet": "b2"}],
    ])

    async def run():
        queries = {
            "task-a": ("alpha market", "beta revenue"),
            "task-b": ("gamma policy", "delta adoption"),
        }
        for task_id, task_queries in queries.items():
            for query in task_queries:
                await controller.execute(
                    tool,
                    {"query": query},
                    context={"task_id": task_id},
                )

    asyncio.run(run())
    snapshot = controller.snapshot()
    assert len(tool.calls) == 4
    assert snapshot["stats"]["backend_calls"] == 4
    assert snapshot["task_backend_calls"] == {"task-a": 2, "task-b": 2}


def test_policy_mechanism_requires_candidate_behavior_change():
    def detail(group, action):
        return {
            "pair_id": "q:r1",
            "group": group,
            "mechanism_triggered": True,
            "telemetry": {"search": {"policy_decisions": [
                {"rule_id": "rule", "action": action},
            ]}},
        }

    unchanged = [detail("baseline", "accept_results"), detail("candidate", "accept_results")]
    _mark_policy_mechanism_changes(unchanged)
    assert not any(item["mechanism_triggered"] for item in unchanged)

    changed = [detail("baseline", "accept_results"), detail("candidate", "rewrite_uncovered_facets")]
    _mark_policy_mechanism_changes(changed)
    assert changed[0]["mechanism_triggered"] is False
    assert changed[1]["mechanism_triggered"] is True
