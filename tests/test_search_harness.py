"""Focused tests for search control and reliable tool execution."""
from __future__ import annotations

import asyncio
import json

import pytest

from src.tools.execution_policy import ToolExecutionPolicy
from src.tools.search_controller import SearchController
from src.agents.researcher import ResearcherAgent
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


class _Search:
    name = "web_search"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def execute(self, query: str, top_n: int = 5):
        self.calls.append(query)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"query": query, "results": response, "total": len(response)}


def test_search_controller_caches_and_deduplicates_across_calls():
    tool = _Search([
        [{"title": "A", "url": "https://www.example.com/a?utm_source=x", "snippet": "one"}],
    ])
    controller = SearchController()
    async def run():
        first = await controller.execute(tool, {"query": "Transformer investment", "top_n": 5})
        second = await controller.execute(tool, {"query": "Transformer investment", "top_n": 5})
        return first, second
    first, second = asyncio.run(run())

    assert len(tool.calls) == 1
    assert first["evidence_novelty"] == 1.0
    assert second["cache_hit"] is True
    assert second["evidence_novelty"] == 0.0
    snapshot = controller.snapshot()
    assert snapshot["stats"]["cache_hits"] == 1
    assert len(snapshot["events"]) == 2
    assert snapshot["events"][0]["result_count"] == 1
    assert snapshot["events"][1]["cache_hit"] is True
    assert snapshot["events"][0]["query_fingerprint"]
    assert "Transformer investment" not in json.dumps(snapshot["events"])


def test_search_controller_rewrites_similar_queries_with_hints():
    tool = _Search([
        [{"title": "A", "url": "https://example.com/a", "snippet": "one"}],
        [{"title": "B", "url": "https://example.com/b", "snippet": "two"}],
    ])
    controller = SearchController(query_similarity_threshold=0.6)
    async def run():
        await controller.execute(tool, {"query": "AI funding", "top_n": 5})
        return await controller.execute(
            tool,
            {"query": "AI funding latest", "top_n": 5},
            context={"search_hints": ["regional breakdown"]},
        )
    result = asyncio.run(run())

    assert len(tool.calls) == 2
    assert tool.calls[1] != "AI funding latest"
    assert result["rewritten_query"] == tool.calls[1]


def test_search_event_retains_strategy2_frontier_metadata():
    tool = _Search([[{"title": "A", "url": "https://example.com/a", "snippet": "one"}]])
    controller = SearchController()
    asyncio.run(controller.execute(
        tool,
        {"query": "evidence"},
        context={
            "facet_id": "safety",
            "claim_ids": ["claim_1"],
            "source_cluster_ids": ["domain:example.com"],
            "action": "cross_validate",
            "estimated_value": 0.7,
        },
    ))
    event = controller.snapshot()["events"][0]
    assert event["facet_id"] == "safety"
    assert event["claim_ids"] == ["claim_1"]
    assert event["action"] == "cross_validate"
    assert event["estimated_value"] == 0.7


def test_search_controller_rewrites_an_empty_first_query():
    tool = _Search([
        [],
        [{"title": "A", "url": "https://example.com/a", "snippet": "found"}],
    ])
    controller = SearchController()

    async def run():
        return await controller.execute(tool, {"query": "rare query"})

    result = asyncio.run(run())
    assert len(tool.calls) == 2
    assert result["results"]
    assert result["rewritten_query"] == tool.calls[1]


class _Flaky:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    async def execute(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("temporary timeout")
        return {"ok": True}


def test_execution_policy_retries_transient_errors():
    tool = _Flaky()
    policy = ToolExecutionPolicy(max_retries=2, retry_delay=0, circuit_failure_threshold=5)
    result = asyncio.run(policy.execute("flaky", tool, {}))
    assert result["ok"] is True
    assert tool.calls == 3
    assert len(result["_tool_execution"]["attempts"]) == 3


class _AlwaysFail:
    name = "primary"

    async def execute(self, **kwargs):
        raise RuntimeError("HTTP 503 service unavailable")


class _Fallback:
    name = "fallback"

    async def execute(self, **kwargs):
        return {"value": "fallback"}


def test_execution_policy_falls_back_after_server_error():
    policy = ToolExecutionPolicy(max_retries=0, retry_delay=0, circuit_failure_threshold=1)
    result = asyncio.run(policy.execute(
        "primary",
        _AlwaysFail(),
        {"query": "x"},
        fallback_tools=[("fallback", _Fallback(), {"query": "x"})],
    ))
    assert result["value"] == "fallback"
    assert result["_tool_execution"]["selected_tool"] == "fallback"


class _Policy:
    def __init__(self):
        self.turn = 0

    def set_tools(self, schemas):
        self.schemas = schemas

    def __call__(self, messages):
        self.turn += 1
        if self.turn == 1:
            return {"content": "answer without evidence", "tool_calls": []}
        if self.turn == 2:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "test"})},
                }],
            }
        return {"content": "evidence-backed answer Confidence: 0.8", "tool_calls": []}


class _OneResultSearch:
    name = "web_search"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    async def execute(self, query: str, top_n: int = 5):
        return {"query": query, "results": [{"title": "T", "url": "https://example.com/t", "snippet": "evidence"}]}


class _Arxiv:
    name = "arxiv_reader"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    async def execute(self, query: str, max_results: int = 3):
        return {"papers": [{"title": "Paper", "pdf_url": "https://arxiv.org/pdf/1", "summary": "paper evidence"}]}


class _Browser:
    name = "browser"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    async def execute(self, url: str, max_chars: int = 8000):
        return "full page evidence"


def test_researcher_requires_tool_before_success():
    agent = ResearcherAgent("r", _Policy(), [_OneResultSearch()])
    result = asyncio.run(agent.run(SubTask("t", TaskType.SEARCH, "test"), {"query": "test"}))
    assert result.status is AgentStatus.SUCCESS
    assert any(item.get("event") == "tool_required" for item in result.trajectory)
    assert result.evidence_bundle["sources"][0]["source_id"].startswith("src_")


def test_researcher_prompt_includes_dependency_evidence():
    agent = ResearcherAgent("r", _Policy(), [])
    task = SubTask(
        "child",
        TaskType.ANALYZE,
        "compare evidence",
        dependencies=["parent"],
    )
    prompt = agent._build_task_prompt(
        task,
        {
            "dep:parent": AgentResult(
                "parent", AgentStatus.SUCCESS, output="primary-source finding", confidence=0.8
            )
        },
    )
    assert "Dependency evidence" in prompt
    assert "primary-source finding" in prompt


def test_researcher_prompt_uses_runtime_tool_budget() -> None:
    agent = ResearcherAgent("r", _Policy(), [], max_tool_calls=6)
    task = SubTask("t", TaskType.SEARCH, "test")
    assert "AT MOST 6" in agent._system_prompt()
    assert "AT MOST 6" in agent._build_task_prompt(task, {"query": "test"})
    assert "AT MOST 2" not in agent._system_prompt()


def test_non_web_evidence_does_not_trigger_false_stop() -> None:
    class Policy:
        def __init__(self):
            self.turn = 0

        def set_tools(self, _schemas):
            pass

        def __call__(self, messages):
            self.turn += 1
            if self.turn == 1:
                return {"content": "", "tool_calls": [{
                    "id": "a", "function": {"name": "arxiv_reader", "arguments": '{"query":"paper"}'},
                }]}
            if self.turn == 2:
                assert not any("Write your final summary NOW" in str(m.get("content", "")) for m in messages)
                return {"content": "", "tool_calls": [{
                    "id": "b", "function": {"name": "browser", "arguments": '{"url":"https://arxiv.org/pdf/1"}'},
                }]}
            return {"content": "supported answer Confidence: 0.8", "tool_calls": []}

    agent = ResearcherAgent("r", Policy(), [_Arxiv(), _Browser()], max_tool_calls=4)
    result = asyncio.run(agent.run(SubTask("t", TaskType.SEARCH, "paper research"), {"query": "paper"}))
    assert result.status is AgentStatus.SUCCESS
    assert not any(step.get("event") == "evidence_gain_stop" for step in result.trajectory)


def test_tool_budget_stop_summarizes_instead_of_failing() -> None:
    class Policy:
        def __init__(self):
            self.turn = 0

        def set_tools(self, _schemas):
            pass

        def __call__(self, _messages):
            self.turn += 1
            if self.turn <= 2:
                return {"content": "", "tool_calls": [{
                    "id": str(self.turn),
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }]}
            return {"content": "bounded summary Confidence: 0.7", "tool_calls": []}

    agent = ResearcherAgent("r", Policy(), [_OneResultSearch()], max_tool_calls=1)
    result = asyncio.run(agent.run(SubTask("t", TaskType.SEARCH, "test"), {"query": "test"}))
    assert result.status is AgentStatus.SUCCESS
    assert any(
        isinstance(step.get("result"), dict) and step["result"].get("error_type") == "budget"
        for step in result.trajectory
    )


def test_search_controller_hard_cap_exists_without_evolution_policy() -> None:
    tool = _Search([
        [{"title": "A", "url": "https://example.com/a", "snippet": "a"}],
        [{"title": "must-not-run", "url": "https://example.com/b", "snippet": "b"}],
    ])
    controller = SearchController(max_backend_calls=1, max_rewrites=0)

    async def run():
        await controller.execute(tool, {"query": "first"})
        return await controller.execute(tool, {"query": "second"})

    blocked = asyncio.run(run())
    assert tool.calls == ["first"]
    assert blocked["hard_cap_reached"] is True


def test_verify_output_contract_fails_closed_without_evidence_span() -> None:
    task = SubTask("verify_1", TaskType.VERIFY, "claim_id=c1 verify revenue")
    output = ResearcherAgent._normalise_verify_output(
        '{"claim_id":"c1","status":"supported","evidence":[],"confidence":0.9}',
        task,
    )
    payload = json.loads(output)
    assert payload["status"] == "unknown"
    assert payload["reason"] == "supported_without_attributable_span"

    agent = ResearcherAgent("r", _Policy(), [], max_tool_calls=3)
    assert "REQUIRED VERIFY OUTPUT CONTRACT" in agent._build_task_prompt(task, {"query": "q"})
