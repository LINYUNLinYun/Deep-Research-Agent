from __future__ import annotations

import asyncio
import json
import time

from src.adversarial.blue_agent import BlueAgent
from src.adversarial.loop import AdversarialLoop
from src.adversarial.red_agent import RedAgent
from src.adversarial.verdict import Dimension, FixOperation, FixType, Issue, RedVerdict, Severity
from src.evidence import EvidenceVerifier, VerificationStatus
from src.tools.search_controller import SearchController
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import OrchestratorState, ResearchReport, RunConfig


def test_evidence_verifier_supports_number_without_counting_citation_marker() -> None:
    report = {
        "content": "The growth rate reached 20% in 2024 [1].",
        "sources": [
            {
                "url": "https://example.com/stats",
                "title": "Official statistics",
                "snippet": "The growth rate reached 20% in 2024.",
            }
        ],
    }
    result = EvidenceVerifier().verify_sync(report)[0]
    assert result.status is VerificationStatus.SUPPORTED
    assert result.evidence[0].claim_id == result.claim_id


def test_evidence_verifier_is_conservative_without_source_span() -> None:
    report = {"content": "The system has a 99% success rate.", "sources": [{"url": "https://x.test"}]}
    result = EvidenceVerifier().verify_sync(report)[0]
    assert result.status is VerificationStatus.UNKNOWN


def test_evidence_verifier_resolves_explicit_citation_id_and_source_id() -> None:
    report = {"content": "Revenue reached 20% in 2024 [7]."}
    sources = [{
        "citation_id": 7,
        "source_id": "src_official",
        "url": "https://example.com/stats",
        "source_span": "Revenue reached 20% in 2024.",
    }]
    result = EvidenceVerifier().verify_sync(report, sources)[0]
    assert result.status is VerificationStatus.SUPPORTED
    assert result.evidence[0].metadata["source_id"] == "src_official"


def test_blue_merge_preserves_unseen_report_suffix() -> None:
    blue = BlueAgent(policy=lambda _: {"content": ""})
    original = "HEAD\n" + ("middle\n" * 900) + "TAIL"
    candidate = "HEAD\ncorrected excerpt"
    merged = blue._merge_fixed_content(original, candidate, [])
    assert merged == original
    patched = blue._merge_fixed_content(
        original,
        candidate,
        [{"before": "TAIL", "after": "FINAL"}],
    )
    assert patched.endswith("FINAL")
    assert "middle" in patched
    patch_only = blue._merge_fixed_content(
        original,
        "",
        [{"before": "TAIL", "after": "PATCHED"}],
    )
    assert patch_only.endswith("PATCHED")


def test_adversarial_context_budgets_are_configurable_and_bounded() -> None:
    red = RedAgent(policy=lambda _: {}, context_chars=9000)
    blue = BlueAgent(
        policy=lambda _: {},
        repair_context_chars=10000,
        self_verify_context_chars=7000,
    )
    content = "x" * 20000
    assert len(red._excerpt(content, red.context_chars)) == 9000
    assert len(blue._truncate_content(content)) == 10000
    assert blue.self_verify_context_chars == 7000

    capped_red = RedAgent(policy=lambda _: {}, context_chars=100000)
    capped_blue = BlueAgent(policy=lambda _: {}, repair_context_chars=100000)
    assert capped_red.context_chars == 32000
    assert capped_blue.repair_context_chars == 32000


def test_blue_issue_context_centers_long_report_on_exact_claim() -> None:
    claim = "关键事实显示该指标达到42%，但来源并未支持"
    content = (
        "HEAD\n" + ("intro filler " * 180) + "\n\n"
        "## 核心章节\n\n前一段背景。\n\n" + claim + "[7]。\n\n后一段分析。\n\n"
        + ("tail filler " * 180) + "\nTAIL"
    )
    sources = [{"title": str(i), "url": f"https://{i}.test", "snippet": "s"} for i in range(1, 8)]
    issue = Issue(
        Severity.MAJOR,
        Dimension.FACTUAL,
        f"报告中的“{claim}”与来源不一致",
        "核心章节",
        FixType.IN_PLACE,
    )
    context = BlueAgent(policy=lambda _: {}, repair_context_chars=1800)._build_issue_context(
        ResearchReport("q", content, sources), issue
    )
    assert context.matched_by == "exact_claim"
    assert claim in context.target_text
    assert f"<TARGET>\n{context.target_text}\n</TARGET>" in context.excerpt
    assert "HEAD" in context.excerpt
    assert "TAIL" in context.excerpt
    assert context.source_ids == (7,)
    assert len(context.excerpt) <= 1800


def test_blue_issue_context_matches_model_authored_section_location() -> None:
    content = (
        "## 执行摘要\nsummary\n\n"
        "### 2.1 硬件平台与出货量\nwrong section\n\n"
        "### 2.2 软件与操作栈：原生优先\ntarget section claim\n\n"
        "### 2.3 内容创作\nnext section"
    )
    issue = Issue(
        Severity.MAJOR,
        Dimension.FACTUAL,
        "相关发布状态需要修正",
        "执行摘要及 2.2 节“苹果侧”",
        FixType.IN_PLACE,
    )
    context = BlueAgent(policy=lambda _: {})._build_issue_context(
        ResearchReport("q", content), issue
    )
    assert context.matched_by == "heading"
    assert "2.2 软件与操作栈" in context.target_text
    assert "target section claim" in context.target_text
    assert "2.3 内容创作" not in context.target_text


def test_blue_issue_context_does_not_target_paper_title_in_references() -> None:
    content = (
        "## 正文\n\n正文中的保险定价论述[1]。\n\n"
        "## 元信息\nmetadata\n\n## 参考来源\n"
        "[1] 《AI Liability Insurance With an Example》"
    )
    issue = Issue(
        Severity.MAJOR,
        Dimension.FACTUAL,
        "报告虚构了论文《AI Liability Insurance With an Example》的内容",
        "正文",
        FixType.IN_PLACE,
    )
    context = BlueAgent(policy=lambda _: {})._build_issue_context(
        ResearchReport("q", content), issue
    )
    assert context.matched_by == "heading"
    assert "正文中的保险定价论述" in context.target_text
    assert "参考来源" not in context.target_text


def test_blue_merge_rejects_patch_outside_or_ambiguous_in_target() -> None:
    blue = BlueAgent(policy=lambda _: {})
    original = "outside sentence\n\ntarget sentence\n\nending"
    start = original.index("target sentence")
    end = start + len("target sentence")
    assert blue._merge_fixed_content(
        original,
        "",
        [{"before": "outside sentence", "after": "changed outside"}],
        target_range=(start, end),
    ) == original
    assert "fixed target" in blue._merge_fixed_content(
        original,
        "",
        [{"before": "target sentence", "after": "fixed target"}],
        target_range=(start, end),
    )

    repeated = "same sentence\n\nsame sentence"
    assert blue._merge_fixed_content(
        repeated,
        "",
        [{"before": "same sentence", "after": "changed"}],
        target_range=(0, len("same sentence")),
    ) == repeated


def test_blue_self_verify_handles_dimension_without_name_error() -> None:
    class Policy:
        def __call__(self, _messages):
            return {"content": json.dumps({"has_new_issue": True, "new_issues": [{"description": "x"}]})}

    async def run():
        return await BlueAgent(Policy())._self_verify("old", "new", [])

    ok, issues = asyncio.run(run())
    assert not ok
    assert issues and issues[0].dimension is Dimension.LOGICAL


def test_blue_self_verify_accepts_fenced_json() -> None:
    class Policy:
        def __call__(self, _messages):
            return {"content": '```json\n{"has_new_issue": false, "new_issues": []}\n```'}

    async def run():
        return await BlueAgent(Policy())._self_verify("old", "new", [])

    ok, issues = asyncio.run(run())
    assert ok
    assert issues == []


def test_blue_supplementary_search_uses_shared_controller_budget() -> None:
    class Search:
        name = "web_search"

        def __init__(self):
            self.calls = 0

        async def execute(self, query: str, top_n: int = 5):
            self.calls += 1
            return {"results": [{
                "title": "Official", "url": "https://example.com/official", "snippet": "supported",
            }]}

    search = Search()
    controller = SearchController(max_backend_calls=1, max_rewrites=0)
    blue = BlueAgent(
        policy=lambda _messages: {"content": "{}"},
        tools=[search],
        search_controller=controller,
    )
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "verify this claim", "p1", FixType.SUPPLEMENTARY)
    report = ResearchReport("q", "claim", [])

    async def run():
        await blue._do_supplementary_search(report, issue)
        second_issue = Issue(
            Severity.MAJOR, Dimension.FACTUAL, "verify another claim", "p2", FixType.SUPPLEMENTARY
        )
        return await blue._do_supplementary_search(report, second_issue)

    second = asyncio.run(run())
    assert search.calls == 1
    assert second.action == "supplementary_search_failed"
    events = controller.snapshot()["events"]
    assert [event["stage"] for event in events] == ["blue", "blue"]
    assert events[-1]["hard_cap_reached"] is True


def test_adversarial_loop_final_score_is_post_fix_score() -> None:
    class Red:
        def __init__(self):
            self.calls = 0

        async def attack(self, report):
            self.calls += 1
            if self.calls == 1:
                issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "bad", "p1", FixType.IN_PLACE)
                return RedVerdict({Dimension.FACTUAL: 4.0}, 4.0, [issue])
            return RedVerdict({Dimension.FACTUAL: 9.0}, 9.0, [])

    class Blue:
        status = "success"
        error = ""

        async def defend(self, report, verdict):
            fixed = ResearchReport(report.query, report.content + " fixed", report.sources)
            return fixed, [FixOperation(verdict.issues[0], "fix", True)]

    async def run():
        return await AdversarialLoop(Red(), Blue(), max_rounds=2, score_threshold=8).run(
            ResearchReport("q", "old")
        )

    report, history = asyncio.run(run())
    assert report.content.endswith("fixed")
    assert report.final_score == 9.0
    assert history[0]["pre_fix_score"] == 4.0
    assert history[0]["post_fix_score"] == 9.0


def test_red_agent_deduplicates_and_caps_issues() -> None:
    class Policy:
        def __call__(self, _messages):
            issue = {
                "severity": "major",
                "description": "duplicate",
                "location": "p1",
                "fix_type": "in_place",
            }
            return {"content": json.dumps({"score": 4, "issues": [issue, issue]})}

    async def run():
        agent = RedAgent(Policy(), max_issues=2)
        verdict = await agent.attack(ResearchReport("q", "content"))
        return agent, verdict

    agent, verdict = asyncio.run(run())
    assert len(verdict.issues) == 2
    assert agent.last_issue_stats == {"raw": 10, "deduplicated": 5, "selected": 2}
    assert verdict.issue_stats == agent.last_issue_stats


def test_red_agent_prompt_bounds_dimension_issue_output() -> None:
    class Policy:
        def __init__(self):
            self.prompts = []

        def __call__(self, messages):
            self.prompts.append(messages[-1]["content"])
            return {"content": json.dumps({"score": 8, "issues": []})}

    async def run():
        policy = Policy()
        await RedAgent(policy, max_issues_per_dimension=2).attack(ResearchReport("q", "content"))
        return policy

    policy = asyncio.run(run())
    assert len(policy.prompts) == 5
    assert all("最多 2 个" in prompt for prompt in policy.prompts)


def test_red_agent_retries_only_the_malformed_dimension() -> None:
    class Policy:
        def __init__(self):
            self.calls = 0

        def __call__(self, _messages):
            self.calls += 1
            if self.calls == 3:
                return {"content": "not-json"}
            return {"content": json.dumps({"score": 8, "issues": []})}

    async def run():
        policy = Policy()
        agent = RedAgent(policy, dimension_parse_retries=1)
        verdict = await agent.attack(ResearchReport("q", "content"))
        return policy, agent, verdict

    policy, agent, verdict = asyncio.run(run())
    assert verdict.status == "success"
    assert policy.calls == 6  # five dimensions plus one targeted retry
    assert agent.last_retry_stats == {"attempted": 1, "recovered": 1, "exhausted": 0}
    assert verdict.retry_stats == agent.last_retry_stats


def test_adversarial_loop_records_failed_red_attempt() -> None:
    class Red:
        async def attack(self, report):
            return RedVerdict(
                {Dimension.FACTUAL: 0}, 0, [], "invalid output",
                status="failed", error="invalid JSON",
            )

    class Blue:
        async def defend(self, report, verdict):
            raise AssertionError("Blue must not run after failed Red")

    async def run():
        return await AdversarialLoop(Red(), Blue()).run(ResearchReport("q", "old"))

    report, history = asyncio.run(run())
    assert report.adversarial_status == "skipped"
    assert report.adversarial_rounds == 1
    assert len(history) == 1
    assert history[0]["accepted"] is False
    assert history[0]["outcome"] == "failed"


def test_adversarial_loop_keeps_best_report_after_later_red_failure() -> None:
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "bad", "p1", FixType.IN_PLACE)

    class Red:
        def __init__(self):
            self.calls = 0

        async def attack(self, report):
            self.calls += 1
            if self.calls == 1:
                return RedVerdict({Dimension.FACTUAL: 4}, 4, [issue])
            if self.calls == 2:
                return RedVerdict({Dimension.FACTUAL: 9}, 9, [])
            return RedVerdict(
                {Dimension.FACTUAL: 0}, 0, [], status="failed", error="invalid JSON"
            )

    class Blue:
        status = "success"
        error = ""

        async def defend(self, report, verdict):
            fixed = ResearchReport(report.query, report.content + " fixed", report.sources)
            return fixed, [FixOperation(issue, "fix", True)]

    async def run():
        return await AdversarialLoop(
            Red(), Blue(), max_rounds=2, score_threshold=10
        ).run(ResearchReport("q", "old"))

    report, history = asyncio.run(run())
    assert report.content == "old fixed"
    assert report.final_score == 9
    assert report.adversarial_status == "partial"
    assert report.adversarial_rounds == 2
    assert history[-1]["fallback_to_best"] is True


def test_adversarial_loop_keeps_best_report_after_later_blue_failure() -> None:
    first = Issue(Severity.MAJOR, Dimension.FACTUAL, "first", "p1", FixType.IN_PLACE)
    second = Issue(Severity.MAJOR, Dimension.LOGICAL, "second", "p2", FixType.IN_PLACE)

    class Red:
        def __init__(self):
            self.calls = 0

        async def attack(self, report):
            self.calls += 1
            if self.calls == 1:
                return RedVerdict({Dimension.FACTUAL: 4}, 4, [first])
            if self.calls == 2:
                return RedVerdict({Dimension.FACTUAL: 9}, 9, [])
            return RedVerdict({Dimension.FACTUAL: 8}, 8, [second])

    class Blue:
        error = ""

        def __init__(self):
            self.calls = 0
            self.status = "success"

        async def defend(self, report, verdict):
            self.calls += 1
            if self.calls == 1:
                return ResearchReport(report.query, report.content + " fixed", report.sources), [
                    FixOperation(first, "fix", True)
                ]
            self.status = "failed"
            self.error = "second round failed"
            return report, [FixOperation(second, "fix", False)]

    async def run():
        return await AdversarialLoop(
            Red(), Blue(), max_rounds=2, score_threshold=10
        ).run(ResearchReport("q", "old"))

    report, history = asyncio.run(run())
    assert report.content == "old fixed"
    assert report.final_score == 9
    assert report.adversarial_status == "partial"
    assert report.adversarial_reason == "second round failed"
    assert history[-1]["fallback_to_best"] is True


def test_blue_agent_rolls_back_only_unverified_fix() -> None:
    issues = [
        Issue(Severity.MAJOR, Dimension.FACTUAL, "one", "p1", FixType.IN_PLACE),
        Issue(Severity.MAJOR, Dimension.LOGICAL, "two", "p2", FixType.IN_PLACE),
        Issue(Severity.MINOR, Dimension.COVERAGE, "three", "p3", FixType.IN_PLACE),
    ]

    class Blue(BlueAgent):
        def __init__(self):
            super().__init__(policy=lambda _: {})
            self.verifications = 0

        async def _fix_single_issue(self, report, issue):
            report.content += f"|{issue.description}"
            return FixOperation(issue, "fix", True)

        async def _self_verify(self, original, revised, operations):
            self.verifications += 1
            if self.verifications != 2:
                return True, []
            return False, [Issue(Severity.MAJOR, Dimension.LOGICAL, "new", "p3")]

    async def run():
        blue = Blue()
        report, operations = await blue.defend(
            ResearchReport("q", "old"),
            RedVerdict({Dimension.FACTUAL: 4}, 4, issues),
        )
        return blue, report, operations

    blue, report, operations = asyncio.run(run())
    assert report.content == "old|one|three"
    assert blue.status == "partial"
    assert operations[0].success is True
    assert operations[1].success is False
    assert "rolled_back_after_self_verify" in operations[1].detail
    assert operations[3].success is True
    assert blue.last_repair_stats == {
        "selected": 3,
        "attempted": 3,
        "committed": 2,
        "rolled_back": 1,
        "skipped_after_failure_cap": 0,
    }


def test_blue_agent_stops_after_bounded_consecutive_failures() -> None:
    issues = [
        Issue(Severity.MAJOR, Dimension.FACTUAL, name, name, FixType.IN_PLACE)
        for name in ("one", "two", "three")
    ]

    class Blue(BlueAgent):
        def __init__(self):
            super().__init__(policy=lambda _: {}, max_consecutive_failures=2)

        async def _fix_single_issue(self, report, issue):
            return FixOperation(issue, "candidate_rejected", False, issue.description)

    async def run():
        blue = Blue()
        report, operations = await blue.defend(
            ResearchReport("q", "old"),
            RedVerdict({Dimension.FACTUAL: 4}, 4, issues),
        )
        return blue, report, operations

    blue, report, operations = asyncio.run(run())
    assert report.content == "old"
    assert len(operations) == 2
    assert blue.status == "failed"
    assert blue.last_repair_stats["skipped_after_failure_cap"] == 1


def test_blue_supplementary_sources_are_bound_to_report_citations() -> None:
    blue = BlueAgent(policy=lambda _: {})
    report = ResearchReport(
        "q",
        "Claim before [1].",
        [{"title": "Existing", "url": "https://old.test", "snippet": "old"}],
    )
    candidates = [{"title": "New evidence", "url": "https://new.test", "snippet": "supports claim"}]
    changes, additions, references, error = blue._prepare_supplementary_changes(
        report,
        [{"before": "Claim before [1].", "after": "Claim after {{SOURCE_1}}."}],
        candidates,
    )
    assert error == ""
    assert changes[0]["after"] == "Claim after [2]."
    assert additions == candidates
    assert references[0].startswith("[2] [New evidence](https://new.test)")

    bracketed, _, _, error = blue._prepare_supplementary_changes(
        report,
        [{"before": "Claim before [1].", "after": "Claim after [{{SOURCE_1}}]."}],
        candidates,
    )
    assert error == ""
    assert bracketed[0]["after"] == "Claim after [2]."


def test_blue_rejects_unbound_numeric_citation_from_supplementary_fix() -> None:
    blue = BlueAgent(policy=lambda _: {})
    report = ResearchReport("q", "Claim [1].", [{"url": "https://old.test"}])
    prepared = blue._prepare_supplementary_changes(
        report,
        [{"before": "Claim [1].", "after": "Claim [99]."}],
        [{"title": "New", "url": "https://new.test", "snippet": "evidence"}],
    )
    assert prepared[0] == []
    assert "[99]" in prepared[3]


def test_blue_self_verify_context_focuses_on_changed_region() -> None:
    original = "HEAD " + ("unrelated " * 1000) + "old fact" + (" tail" * 1000)
    revised = original.replace("old fact", "new fact")
    before, after = BlueAgent._change_contexts(original, revised)
    assert "old fact" in before
    assert "new fact" in after
    assert "HEAD" not in before
    assert len(before) <= 4000

    with_source = revised + "\n\n### 补充来源\n[2] [Source](https://source.test) — evidence"
    _, after_with_source = BlueAgent._change_contexts(original, with_source)
    assert "### 补充来源" in after_with_source
    assert "https://source.test" in after_with_source


def test_adversarial_loop_keeps_verified_partial_blue_result() -> None:
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "bad", "p1", FixType.IN_PLACE)

    class Red:
        async def attack(self, report):
            score = 9.0 if report.content.endswith("fixed") else 4.0
            return RedVerdict({Dimension.FACTUAL: score}, score, [] if score > 4 else [issue])

    class Blue:
        status = "partial"
        error = "second repair failed"

        async def defend(self, report, verdict):
            return ResearchReport(report.query, report.content + " fixed", report.sources), [
                FixOperation(issue, "fix", True)
            ]

    async def run():
        return await AdversarialLoop(Red(), Blue(), max_rounds=2).run(
            ResearchReport("q", "old")
        )

    report, history = asyncio.run(run())
    assert report.content == "old fixed"
    assert report.adversarial_status == "partial"
    assert report.final_score == 9.0
    assert history[0]["accepted"] is True
    assert history[0]["outcome"] == "partial"


def test_adversarial_loop_rejects_post_fix_score_regression() -> None:
    issue = Issue(Severity.MAJOR, Dimension.FACTUAL, "bad", "p1", FixType.IN_PLACE)

    class Red:
        async def attack(self, report):
            score = 4.0 if report.content.endswith("worse") else 5.0
            return RedVerdict({Dimension.FACTUAL: score}, score, [issue])

    class Blue:
        status = "success"
        error = ""

        async def defend(self, report, verdict):
            return ResearchReport(report.query, report.content + " worse", report.sources), [
                FixOperation(issue, "fix", True)
            ]

    async def run():
        return await AdversarialLoop(Red(), Blue(), max_rounds=1).run(
            ResearchReport("q", "old")
        )

    report, history = asyncio.run(run())
    assert report.content == "old"
    assert report.adversarial_status == "rejected"
    assert report.final_score == 5.0
    assert history[0]["accepted"] is False
    assert history[0]["outcome"] == "discarded"


def test_orchestrator_adversarial_stage_has_hard_timeout() -> None:
    class SlowLoop:
        async def run(self, report):
            await asyncio.sleep(1)
            return report, []

    async def run():
        orchestrator = Orchestrator.__new__(Orchestrator)
        report = ResearchReport("q", "old", confidence=0.1)
        orchestrator._memory_store = {"final_report": report}
        orchestrator._config = RunConfig(
            global_timeout_seconds=10,
            adversarial_timeout_seconds=0.01,
            adversarial_confidence_threshold=0.8,
        )
        orchestrator._start_time = time.monotonic()
        orchestrator._adversarial_count = 0
        orchestrator.adversarial_loop = SlowLoop()
        state = await orchestrator._do_adversarial()
        return state, report

    state, report = asyncio.run(run())
    assert state is OrchestratorState.DONE
    assert report.adversarial_status == "skipped"
    assert report.adversarial_reason == "adversarial_timeout"
