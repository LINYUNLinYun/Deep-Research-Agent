import copy
import json
import asyncio

from src.agents.researcher import ResearcherAgent
from src.compressor.compressor import ContextCompressor
from src.models.vllm_policy import VLLMPolicy
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


def tool_group(prefix, count=4, size=12000):
    calls = [{"id": f"{prefix}{i}", "type": "function", "function": {
        "name": "web_search", "arguments": "{}",
    }} for i in range(count)]
    return [{"role": "assistant", "content": "", "tool_calls": calls}] + [
        {"role": "tool", "tool_call_id": call["id"], "content": "证据" * size}
        for call in calls
    ]


def test_parallel_tool_group_does_not_erase_task_or_evidence():
    policy = object.__new__(VLLMPolicy)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "original task"}] + tool_group("new")
    original = copy.deepcopy(messages)
    result = policy._truncate_messages(messages)
    assert result[1] == original[1]
    assert result[2]["tool_calls"] == original[2]["tool_calls"]
    assert [m["tool_call_id"] for m in result[3:]] == [f"new{i}" for i in range(4)]
    assert all(m["content"] for m in result[3:])
    assert sum(len(m["content"]) for m in result) < 35000
    assert messages == original


def test_old_groups_are_removed_without_orphaning_latest_tools():
    policy = object.__new__(VLLMPolicy)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task"}] + tool_group("old") + tool_group("new", size=100)
    result = policy._truncate_messages(messages)
    assert result[1]["content"] == "task"
    assert [m["tool_call_id"] for m in result if m["role"] == "tool"] == [f"new{i}" for i in range(4)]


class Compressor:
    enable_multilevel = True
    chars_per_token = 3.5
    available_budget = 13952

    def __init__(self):
        self.calls = []

    def compress(self, texts, **kwargs):
        self.calls.append(kwargs)
        return ["compressed evidence"]


def test_worker_compresses_each_tool_payload_preserving_protocol():
    compressor = Compressor()
    agent = ResearcherAgent("worker", lambda _: {}, compressor=compressor)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task"}] + tool_group("new")
    original = copy.deepcopy(messages)
    result = agent._compress_tool_messages(messages, "query")
    assert len(compressor.calls) == 4
    assert all(call["system_prompt_tokens"] > 0 for call in compressor.calls)
    assert result[:3] == original[:3]
    assert [m["tool_call_id"] for m in result[3:]] == [f"new{i}" for i in range(4)]
    assert len(json.dumps(result, ensure_ascii=False)) < 30000
    assert messages == original


def test_compression_failure_falls_back_without_losing_task():
    class BrokenCompressor(Compressor):
        def compress(self, texts, **kwargs):
            raise RuntimeError("summarizer unavailable")

    agent = ResearcherAgent("worker", lambda _: {}, compressor=BrokenCompressor())
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "task"}] + tool_group("new")
    result = agent._compress_tool_messages(messages, "query")
    assert result == messages
    policy = object.__new__(VLLMPolicy)
    fallback = policy._truncate_messages(result)
    assert fallback[1]["content"] == "task"
    assert len([m for m in fallback if m["role"] == "tool"]) == 4


def test_compression_preserves_stop_search_notice():
    agent = ResearcherAgent("worker", lambda _: {}, compressor=Compressor())
    notice = "\n\n[SYSTEM NOTICE] Write final summary NOW."
    messages = [{"role": "tool", "tool_call_id": "a", "content": "x" * 40000 + notice}]
    result = agent._compress_tool_messages(messages, "query")
    assert result[0]["content"] == "compressed evidence" + notice


def test_compressor_fallback_honors_reserved_worker_budget():
    class Embedder:
        def encode(self, text):
            return [1.0, 0.0]

    compressor = ContextCompressor(lambda _: {}, embedder=Embedder(),
                                   budget=16000, output_reserve=2048)
    # Force the semantic stages to exceed the budget, as a long LLM summary can.
    compressor._l1_filter = lambda texts, query, budget: texts
    compressor._l2_extract = lambda texts, query, budget: texts
    compressor._l3_summarize = lambda texts, query, budget: texts
    result = compressor.compress(["x" * 40000], system_prompt_tokens=13000)
    assert result
    assert compressor.calculate_tokens(result) <= 952


def test_compressed_dependencies_replace_original_context():
    agent = ResearcherAgent("worker", lambda _: {})
    task = SubTask(task_id="t", description="analyze", task_type=TaskType.ANALYZE,
                   dependencies=["dep"], context_keys=["notes"])
    context = {"dep:dep": AgentResult("dep", AgentStatus.SUCCESS, output="RAW DEPENDENCY"),
               "notes": "RAW NOTES", "compressed_context": "SHORT EVIDENCE"}
    prompt = agent._build_task_prompt(task, context)
    assert "SHORT EVIDENCE" in prompt
    assert "RAW DEPENDENCY" not in prompt
    assert "RAW NOTES" not in prompt
    del context["compressed_context"]
    prompt = agent._build_task_prompt(task, context)
    assert "RAW DEPENDENCY" in prompt and "RAW NOTES" in prompt


def test_worker_loop_compresses_before_next_policy_call_and_keeps_raw_evidence():
    compressor = Compressor()
    seen = []

    def policy(messages):
        seen.append(copy.deepcopy(messages))
        if len(seen) == 1:
            return {"content": "", "tool_calls": [{"id": "search1", "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'}}]}
        return {"content": "Findings supported by evidence. Confidence: 0.9"}

    agent = ResearcherAgent("worker", policy, compressor=compressor)
    raw = "raw evidence " * 5000

    async def execute(*args, **kwargs):
        return {"results": [{"url": "https://example.com/source", "snippet": raw}]}

    agent._execute_tool = execute
    task = SubTask(task_id="search", description="research test", task_type=TaskType.SEARCH)
    result = asyncio.run(agent.run(task, {}))
    assert result.status == AgentStatus.SUCCESS
    assert len(compressor.calls) == 1
    tool = next(m for m in seen[1] if m["role"] == "tool")
    assert tool["tool_call_id"] == "search1"
    assert tool["content"] == "compressed evidence"
    assert raw in json.dumps(result.trajectory, ensure_ascii=False)
