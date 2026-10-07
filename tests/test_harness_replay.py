import asyncio
from types import SimpleNamespace

import pytest

from src.harness_evolution.replay import (
    RecordReplayStore,
    RecordedToolError,
    ReplayMismatchError,
    ReplayToolAdapter,
    attach_replay_tools,
)


class Tool:
    name = "web_search"
    description = "test"

    def get_openai_tool_schema(self):
        return {"type": "function", "function": {"name": self.name}}

    async def execute(self, **kwargs):
        raise AssertionError("real backend must not be called")


def test_replay_is_strict_and_does_not_call_backend(tmp_path):
    store = RecordReplayStore()
    store.record("web_search", {"query": "alpha", "top_n": 5}, response={"results": [1]})
    path = tmp_path / "fixture.json"
    store.save(path)
    loaded = RecordReplayStore.load(path)
    tool = ReplayToolAdapter(Tool(), loaded)
    assert asyncio.run(tool.execute(query="alpha", top_n=5)) == {"results": [1]}
    with pytest.raises(ReplayMismatchError):
        asyncio.run(tool.execute(query="beta", top_n=5))


def test_replay_hash_and_per_query_sequences_are_deterministic():
    store = RecordReplayStore()
    store.record("x", {"q": "a"}, response=1)
    store.record("x", {"q": "a"}, response=2)
    digest = store.sha256()
    assert store.replay("x", {"q": "a"}) == 1
    assert store.replay("x", {"q": "a"}) == 2
    store.reset()
    assert store.replay("x", {"q": "a"}) == 1
    assert store.sha256() == digest


def test_recorded_error_is_not_fixture_coverage_mismatch():
    store = RecordReplayStore()
    store.record("web_search", {"query": "alpha"}, error="TimeoutError: frozen timeout")
    with pytest.raises(RecordedToolError):
        store.replay("web_search", {"query": "alpha"})
    with pytest.raises(ReplayMismatchError):
        store.replay("web_search", {"query": "missing"})


def test_attach_replay_tools_replaces_repairer_agent_cached_tool():
    original = Tool()
    repairer = SimpleNamespace(tools=[original], _search_tool=original)
    modules = {
        "tools": [original],
        "adversarial": SimpleNamespace(repairer_agent=repairer),
    }
    store = RecordReplayStore()
    store.record("web_search", {"query": "frozen"}, response={"results": [1]})

    attach_replay_tools(modules, store)

    assert isinstance(repairer._search_tool, ReplayToolAdapter)
    assert repairer.tools == modules["tools"]
    assert asyncio.run(repairer._search_tool.execute(query="frozen")) == {"results": [1]}
    with pytest.raises(ReplayMismatchError):
        asyncio.run(repairer._search_tool.execute(query="not-recorded"))
