import asyncio

from src.compressor.compressor import ContextCompressor
from src.models.model_router import AdaptiveModelPolicy
from src.orchestrator.agent_pool import AgentPool
from src.orchestrator.orchestrator import Orchestrator
from src.evidence import EvidenceVerifier
from src.orchestrator.schemas import AgentResult, AgentStatus, OrchestratorState, ResearchReport, RunConfig, SubTask, TaskType
from src.planner.dag import DAG
from src.planner.planner import PlanParseError, Planner
from src.tools.search_controller import SearchController


class _Policy:
    was_truncated = False

    def __call__(self, _messages):
        return {"content": "ok"}


def test_planner_rejects_duplicate_and_unknown_dependencies():
    planner = Planner(_Policy())
    duplicate = '{"sub_tasks":[{"task_id":"a"},{"task_id":"a"}]}'
    unknown = '{"sub_tasks":[{"task_id":"a","dependencies":["missing"]}]}'
    for raw in (duplicate, unknown):
        try:
            planner._parse_plan(raw)
        except PlanParseError:
            pass
        else:  # pragma: no cover - explicit assertion message
            raise AssertionError("invalid plan was accepted")


class _RecordingAgent:
    def __init__(self, contexts, fail_first=False):
        self.contexts = contexts
        self.fail_first = fail_first
        self.policy = _Policy()
        self.tools = []

    async def run(self, task, context):
        self.contexts[task.task_id] = context
        status = AgentStatus.FAILED if self.fail_first and task.task_id == "a" else AgentStatus.SUCCESS
        return AgentResult(task.task_id, status, output=f"output-{task.task_id}", confidence=0.8)


class _Pool:
    def __init__(self, agent):
        self.agent = agent

    async def get_agent(self, _task_type):
        return self.agent

    async def release_agent(self, _agent):
        return None


class _PlannerStub:
    pass


def _orchestrator_for_dependency_test(fail_first=False):
    contexts = {}
    orch = Orchestrator(_PlannerStub(), _Pool(_RecordingAgent(contexts, fail_first)))
    dag = DAG()
    dag.add_edge("a", "b")
    orch._dag = dag
    orch._task_map = {
        "a": SubTask("a", TaskType.SEARCH, "first"),
        "b": SubTask("b", TaskType.ANALYZE, "second", dependencies=["a"]),
    }
    orch._query = "query"
    orch._config = RunConfig()
    return orch, contexts


def test_dag_layer_commits_results_before_downstream_execution():
    orch, contexts = _orchestrator_for_dependency_test()
    asyncio.run(orch._do_dispatching())
    assert contexts["b"]["dep:a"].output == "output-a"


def test_failed_dependency_blocks_downstream_agent():
    orch, contexts = _orchestrator_for_dependency_test(fail_first=True)
    asyncio.run(orch._do_dispatching())
    assert "b" not in contexts
    result_b = next(result for result in orch._results if result.task_id == "b")
    assert result_b.status == AgentStatus.FAILED
    assert "dependencies" in result_b.output


def test_agent_pool_releases_to_original_type():
    pool = AgentPool(lambda *_: _Policy(), tools_factory=lambda: [])

    async def exercise():
        agent = await pool.get_agent(TaskType.ANALYZE)
        await pool.release_agent(agent)

    asyncio.run(exercise())
    assert pool.get_stats()["analyze"] == {
        "idle": 1, "active": 0, "created": 1, "degraded": 0
    }


def test_dynamic_replan_uses_failure_impact_not_any_failure():
    orch, _ = _orchestrator_for_dependency_test()
    orch._config = RunConfig(replan_failure_ratio=0.75)
    # Leaf failure plus one usable success does not justify an entire replan.
    orch._dag = DAG()
    orch._dag.add_node("good")
    orch._dag.add_node("leaf")
    results = [
        AgentResult("good", AgentStatus.SUCCESS, output="evidence", confidence=0.8),
        AgentResult("leaf", AgentStatus.FAILED, output="error"),
    ]
    assert orch._decide_after_collection(results).action == "synthesize"


def test_evidence_gap_replan_creates_only_bounded_verify_tasks():
    orch, _ = _orchestrator_for_dependency_test()
    orch._config = RunConfig(evidence_replan_max_tasks=2, max_sub_questions=8)
    orch._results = [
        AgentResult("good", AgentStatus.SUCCESS, output="preserved evidence", confidence=0.8)
    ]
    task_ids = orch._prepare_evidence_gap_tasks({
        "unresolved_claims": [
            {"claim_id": "c1", "text": "Revenue was 20% higher.", "status": "contradicted", "risk": "high"},
            {"claim_id": "c2", "text": "The trial ended in 2024.", "status": "unknown", "risk": "high"},
            {"claim_id": "c3", "text": "A third unsupported claim.", "status": "unknown", "risk": "normal"},
        ]
    })
    assert task_ids == ["verify_gap_r1_1", "verify_gap_r1_2"]
    assert len(orch._dag) == 2
    assert all(orch._task_map[task_id].task_type is TaskType.VERIFY for task_id in task_ids)
    assert all("原始研究问题" in orch._task_map[task_id].description for task_id in task_ids)
    assert [result.task_id for result in orch._historical_results] == ["good"]
    assert orch._results == []


def test_synthesis_evidence_gap_bypasses_broad_planner_replan():
    class SynthesisPolicy:
        tools = None

        def __call__(self, _messages):
            return {"content": "Revenue grew 20% in 2024. Overall Confidence: 0.80"}

    orch = Orchestrator(
        _PlannerStub(),
        _Pool(_RecordingAgent({})),
        summarizer_policy=SynthesisPolicy(),
        evidence_verifier=EvidenceVerifier(max_claims=1),
    )
    orch._query = "revenue"
    orch._results = [AgentResult("good", AgentStatus.SUCCESS, output="finding", confidence=0.8)]
    orch._config = RunConfig(
        enable_replan=True,
        max_replan_rounds=1,
        evidence_replan_threshold=0.5,
        evidence_replan_max_tasks=2,
        enable_adversarial=False,
    )
    # Evidence gaps are a correctness gate and must not be suppressed by the
    # retrieval novelty stop signal.
    orch._low_novelty_rounds = orch._config.replan_novelty_patience
    state = asyncio.run(orch._do_synthesizing())
    assert state is OrchestratorState.DISPATCHING
    assert orch._replan_count == 1
    assert list(orch._task_map) == ["verify_gap_r1_1"]
    assert orch._decision_trace[-1].action == "targeted_verify"
    assert orch._memory_store["prior_report"].content.startswith("Revenue grew")


def test_adaptive_model_policy_falls_back_and_records_route():
    class Failed:
        def __call__(self, _messages):
            return {"status": "failed", "error": "503"}

        def set_tools(self, _tools):
            pass

    class Good:
        was_truncated = False

        def __call__(self, _messages):
            return {"content": "answer"}

        def set_tools(self, _tools):
            pass

    policy = AdaptiveModelPolicy(
        [("cheap", Failed), ("fallback", Good)], max_retries=0, failure_cooldown=1
    )
    assert policy([{"role": "user", "content": "simple"}])["content"] == "answer"
    assert [item["status"] for item in policy.decision_trace] == ["failed", "success"]


def test_compressor_rank_and_pack_is_budget_monotonic():
    class Embedder:
        dim = 2

        def encode(self, text):
            return [1.0, 0.0] if "relevant" in text or text == "query" else [0.0, 1.0]

    compressor = ContextCompressor(
        llm_policy=_Policy(), embedder=Embedder(), budget=30, output_reserve=0
    )
    texts = ["relevant " * 20, "irrelevant material " * 20, "relevant fact " * 10]
    compressed = compressor.compress(texts, query="query", level=1)
    assert compressed
    assert compressor.calculate_tokens(compressed) <= compressor.calculate_tokens(texts)


def test_search_controller_reset_scopes_dedup_to_one_run():
    controller = SearchController()
    controller._seen_urls.add("https://example.com/")
    controller._query_history["q"] = "query"
    controller.reset()
    assert controller.snapshot()["seen_urls"] == 0
    assert controller.snapshot()["cached_queries"] == 0


def test_replan_dispatch_reuses_unchanged_successful_task() -> None:
    orch, contexts = _orchestrator_for_dependency_test()
    task = SubTask("a", TaskType.SEARCH, "first")
    prior = AgentResult("a", AgentStatus.SUCCESS, output="preserved", confidence=0.8)
    orch._dag = DAG()
    orch._dag.add_node("a")
    orch._task_map = {"a": task}
    orch._reusable_results[orch._task_signature(task)] = prior
    asyncio.run(orch._do_dispatching())
    assert "a" not in contexts
    assert orch._results[0].output == "preserved"
    assert orch._results[0].trajectory[-1]["event"] == "reused"


def test_synthesis_source_catalog_is_built_before_l3_compression() -> None:
    class SynthesisPolicy:
        tools = None

        def __init__(self):
            self.prompt = ""

        def __call__(self, messages):
            self.prompt = messages[-1]["content"]
            return {"content": "Fact [1]. Overall Confidence: 0.8"}

    class AggregateCompressor:
        def compress(self, _texts, query=""):
            return ["compressed aggregate without trajectory"]

    policy = SynthesisPolicy()
    orch = Orchestrator(
        _PlannerStub(), _Pool(_RecordingAgent({})), compressor=AggregateCompressor(),
        summarizer_policy=policy,
    )
    orch._query = "q"
    orch._config = RunConfig(enable_adversarial=False, enable_replan=False)
    orch._results = [
        AgentResult("a", AgentStatus.SUCCESS, "a", trajectory=[{
            "role": "tool", "name": "web_search", "result": {"results": [{
                "title": "A", "url": "https://example.com/a", "snippet": "support A",
            }]},
        }]),
        AgentResult("b", AgentStatus.SUCCESS, "b", trajectory=[{
            "role": "tool", "name": "web_search", "result": {"results": [{
                "title": "B", "url": "https://example.com/b", "snippet": "support B",
            }]},
        }]),
    ]
    state = asyncio.run(orch._do_synthesizing())
    assert state is OrchestratorState.DONE
    assert "https://example.com/a" in policy.prompt
    assert "https://example.com/b" in policy.prompt
    assert {source["url"] for source in orch._memory_store["final_report"].sources} == {
        "https://example.com/a", "https://example.com/b",
    }


def test_claim_evidence_edges_are_persisted_with_stable_source_ids() -> None:
    orch, _ = _orchestrator_for_dependency_test()
    orch.evidence_verifier = EvidenceVerifier(max_claims=2)
    report = ResearchReport(
        "q",
        "Revenue reached 20% in 2024 [1].",
        sources=[{
            "citation_id": 1,
            "source_id": "src_stats",
            "url": "https://example.com/stats",
            "source_span": "Revenue reached 20% in 2024.",
        }],
        source_catalog=[{
            "citation_id": 1,
            "source_id": "src_stats",
            "url": "https://example.com/stats",
            "source_span": "Revenue reached 20% in 2024.",
        }],
    )
    summary = asyncio.run(orch._verify_report_evidence(report))
    assert summary["supported"] == 1
    assert report.claim_evidence_edges[0]["source_id"] == "src_stats"
    assert report.claim_evidence_edges[0]["relation"] == "supported"
