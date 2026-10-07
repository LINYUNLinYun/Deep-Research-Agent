import asyncio
import copy
import json

from src.agents.summarizer import SummarizerAgent
from src.agents.researcher import ResearcherAgent
from src.adversarial.critic_agent import CriticAgent
from src.adversarial.repairer_agent import RepairerAgent
from src.adversarial.loop import AdversarialLoop
from src.adversarial.verdict import CriticVerdict, Dimension
from src.evidence import EvidenceLedger, EvidenceVerifier
from src.evidence.confidence import calibrate
from src.evidence.projection import remap_citations
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, ResearchReport, RunConfig
import time
from datetime import date
from src.utils.temporal import infer_source_date, temporal_relevance


def test_month_window_excludes_same_year_old_and_future_events():
    today = date(2026, 10, 8)
    assert temporal_relevance("美股近一个月", "2026-09-04", today=today) == "historical_context"
    assert temporal_relevance("美股近一个月", "2026-09-21", today=today) == "current"
    assert temporal_relevance("美股近一个月", "2026-10-09", today=today) == "future_dated"
    assert temporal_relevance("美股近一个月", "2026", today=today) == "unknown"


def test_publication_url_date_beats_bare_year_and_future_forecast_in_snippet():
    assert infer_source_date({"source_date": "2026", "url": "https://news.example/20260904/article.html", "snippet": "Forecast for 2027"}) == "2026-09-04"
    assert infer_source_date({"published_at": "2026年9月4日"}) == "2026-09-04"


def test_search_patch_can_delete_unsupported_fact_without_touching_neighbors():
    repairer = RepairerAgent(lambda _: {})
    report = ResearchReport("q", "Intro.\n\nRevenue rose 27% [1].\n\nConclusion.")
    changes, additions, lines, error = repairer._prepare_supplementary_changes(
        report, [{"before": "Revenue rose 27% [1].", "after": ""}], [])
    assert not error and not additions and not lines
    fixed = repairer._merge_fixed_content(report.content, "", changes)
    assert "27%" not in fixed
    assert fixed.startswith("Intro.") and fixed.endswith("Conclusion.")
    # A missing replacement field is malformed, not an instruction to delete.
    assert repairer._merge_fixed_content(report.content, "", [{"before": "Intro."}]) == report.content


def test_supplementary_source_identity_survives_noncontiguous_citation_ids():
    repairer = RepairerAgent(lambda _: {})
    report = ResearchReport("q", "Revenue needs checking [7].", sources=[
        {"citation_id": 7, "url": "https://example.com/old"}])
    changes, additions, _, error = repairer._prepare_supplementary_changes(report,
        [{"before": report.content, "after": "Revenue rose 20% in 2026 {{SOURCE_1}}."}],
        [{"title": "Revenue", "url": "https://example.com/new", "snippet": "Revenue rose 20% in 2026."}])
    assert not error
    report.content = repairer._merge_fixed_content(report.content, "", changes)
    report.sources.extend(additions)
    result = EvidenceVerifier().verify_sync(report)[0]
    assert result.status.value == "supported"
    assert result.evidence[0].metadata["citation_id"] == 8


def test_number_grouping_and_unrelated_source_negation_do_not_create_conflicts():
    report = ResearchReport("q", "Nasdaq closed at 27,599.79 points [1].",
        sources=[{"citation_id": 1, "url": "https://example.com/close",
                  "source_span": "Nasdaq closed at 27599.79 points. Investors should not assume future returns."}])
    assert EvidenceVerifier().verify_sync(report)[0].status.value == "supported"
    report.content = "Nasdaq closed at 28,599.79 points [1]."
    assert EvidenceVerifier().verify_sync(report)[0].status.value == "contradicted"
    report.content = "Nasdaq did not close at 27,599.79 points [1]."
    assert EvidenceVerifier().verify_sync(report)[0].status.value == "contradicted"


def test_numeric_claim_cannot_be_supported_by_span_with_no_numbers():
    report = ResearchReport("q", "Revenue grew 20% [1].", sources=[{
        "citation_id": 1, "url": "https://example.com/revenue", "source_span": "Revenue grew strongly."}])
    assert EvidenceVerifier().verify_sync(report)[0].status.value == "unknown"


def test_partial_span_missing_event_month_is_unknown_not_opposing_evidence():
    report = ResearchReport("q", "10月2日因9月就业数据疲软，道琼斯指数上涨0.49% [1]。", sources=[{
        "citation_id": 1, "url": "https://example.com/close",
        "source_span": "10月2日受就业数据疲软影响，道琼斯指数上涨0.49%。"}])
    assert EvidenceVerifier().verify_sync(report)[0].status.value == "unknown"


def test_prior_references_follow_identity_not_old_position():
    old = [{"citation_id": 7, "source_id": "ai", "url": "https://example.com/ai"},
           {"citation_id": 8, "source_id": "gone"}]
    new = [{"citation_id": 7, "source_id": "prices"}, {"citation_id": 2, "source_id": "ai"}]
    draft = remap_citations("AI earnings [7]. Other claim [8].", old, new)
    assert "AI earnings [2]" in draft
    assert "[7]" not in draft and "[8]" not in draft


def test_synthesis_does_not_treat_worker_local_refs_as_catalog_refs():
    agent = SummarizerAgent("s", lambda _: {})
    result = AgentResult("task_7", AgentStatus.SUCCESS, output="Profit rose 27% [7].")
    prompt = agent._build_synthesis_prompt("q", [result], [])
    assert "Profit rose 27% [7]" not in prompt
    assert "Profit rose 27%" in prompt


def test_us_source_projection_drops_unrelated_local_cpi():
    result = AgentResult("search", AgentStatus.SUCCESS, trajectory=[{
        "role": "tool", "name": "web_search", "result": {"results": [
            {"url": "https://info.gov.hk/cpi", "title": "Hong Kong CPI", "snippet": "Consumer prices in Hong Kong"},
            {"url": "https://dcd.wuxi.gov.cn/cpi", "title": "无锡居民消费价格指数", "snippet": "无锡CPI"},
            {"url": "https://example.com/market", "title": "美股上涨", "snippet": "美国股市受盈利推动上涨"},
        ]}}])
    sources = SummarizerAgent("s", lambda _: {}).collect_sources("美股近一个月大涨原因", [result])
    assert [s["url"] for s in sources] == ["https://example.com/market"]


def test_exact_verification_span_enriches_previously_seen_url():
    result = AgentResult("verify", AgentStatus.SUCCESS,
        evidence_bundle={"sources": [{"url": "https://example.com/data", "snippet": "CPI preview"}]},
        output=json.dumps({"status": "supported", "evidence": [
            {"url": "https://example.com/data", "span": "US CPI rose 3.4% in August 2026."}]}))
    sources = EvidenceLedger().catalog([result])
    assert len(sources) == 1
    assert "3.4%" in sources[0]["source_span"]
    assert sources[0]["tool_name"] == "verify_result"


def test_gap_claim_is_not_repeated_when_only_citation_number_changes():
    orch = Orchestrator(None, None)
    orch._config = RunConfig(evidence_replan_max_tasks=3)
    summary = {"unresolved_claims": [{"claim_id": "c", "text": "Profit rose 27% [7].", "status": "unknown"}]}
    assert len(orch._prepare_evidence_gap_tasks(summary)) == 1
    summary["unresolved_claims"][0]["text"] = "Profit rose 27% [2]."
    assert orch._prepare_evidence_gap_tasks(summary) == []


def test_confidence_recalibration_is_idempotent_and_evidence_sensitive():
    report = ResearchReport("q", "", confidence=0.8)
    calibrate(report, {"total_claims": 10, "support_rate": 0.2})
    assert report.confidence == 0.48
    calibrate(report, {"total_claims": 10, "support_rate": 0.2})
    assert report.confidence == 0.48
    calibrate(report, {"total_claims": 10, "support_rate": 0.8})
    assert report.confidence == 0.72


def test_critic_dimensions_really_run_concurrently():
    async def run():
        active = 0
        peak = 0
        async def policy(messages):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"content": '{"score": 8, "issues": []}'}
        verdict = await CriticAgent(policy).attack(ResearchReport("q", "report"))
        assert verdict.status == "success"
        assert peak == 5
    asyncio.run(run())


def test_accepted_fix_recalibrates_confidence_from_same_evidence_basis():
    class Critic:
        async def attack(self, report):
            return CriticVerdict({Dimension.FACTUAL: 8}, 8, [])
    class Repairer:
        status = "success"
        async def defend(self, report, verdict):
            fixed = copy.deepcopy(report)
            fixed.content = "Revenue rose 20% in 2026 [1]."
            return fixed, []
    verifier = EvidenceVerifier()
    report = ResearchReport("q", "Revenue rose 50% in 2026 [1].", confidence=0.8,
        sources=[{"citation_id": 1, "url": "https://example.com/data", "source_span": "Revenue rose 20% in 2026."}])
    report.evidence_verification = verifier.summary(verifier.verify_sync(report))
    calibrate(report, report.evidence_verification)
    fixed, history = asyncio.run(AdversarialLoop(Critic(), Repairer(), max_rounds=1, evidence_verifier=verifier).run(report))
    assert fixed.confidence > report.confidence
    assert fixed.evidence_verification["supported"] == 1
    assert history[0]["pre_confidence"] == report.confidence
    assert history[0]["post_confidence"] == fixed.confidence


def test_verify_json_zero_confidence_is_not_defaulted_to_point_six():
    agent = ResearcherAgent("v", lambda _: {})
    assert agent._extract_confidence('{"status":"unknown","confidence":0.0}') == 0


def test_adversarial_timeout_preserves_already_accepted_checkpoint():
    class SlowLoop:
        async def run(self, report):
            self.best_report = copy.deepcopy(report)
            self.best_report.content = "accepted repair"
            self.best_report.confidence = 0.7
            self.best_report.adversarial_rounds = 1
            self.last_history = [{"round": 1, "accepted": True}]
            await asyncio.sleep(1)
    orch = Orchestrator(None, None, adversarial_loop=SlowLoop())
    orch._memory_store["final_report"] = ResearchReport("q", "old", confidence=0.3)
    orch._config = RunConfig(adversarial_timeout_seconds=0.01)
    orch._start_time = time.monotonic()
    asyncio.run(orch._do_adversarial())
    report = orch._memory_store["final_report"]
    assert report.content == "accepted repair"
    assert report.adversarial_status == "partial"
    assert report.confidence == 0.7
    assert report.adversarial_history[0]["accepted"]


def test_loop_checkpoint_keeps_post_fix_score_before_next_round_finishes():
    class Critic:
        calls = 0
        async def attack(self, report):
            self.calls += 1
            if self.calls == 3:
                await asyncio.sleep(1)
            score = 4 if self.calls == 1 else 5
            return CriticVerdict({Dimension.FACTUAL: score}, score, [])
    class Repairer:
        status = "success"
        async def defend(self, report, verdict):
            fixed = copy.deepcopy(report)
            fixed.content = "accepted revision"
            return fixed, []
    loop = AdversarialLoop(Critic(), Repairer(), score_threshold=9)
    async def run():
        try:
            await asyncio.wait_for(loop.run(ResearchReport("q", "old")), timeout=0.02)
        except asyncio.TimeoutError:
            pass
    asyncio.run(run())
    assert loop.best_report.content == "accepted revision"
    assert loop.best_report.final_score == 5
    assert loop.best_report.adversarial_rounds == 1


def test_deadline_before_adversarial_start_does_not_reuse_previous_query():
    class PreviousLoop:
        best_report = ResearchReport("previous query", "stale accepted report", adversarial_rounds=1)
        last_history = [{"round": 1, "accepted": True}]
        async def run(self, report):
            raise AssertionError("expired loop must not run")
    orch = Orchestrator(None, None, adversarial_loop=PreviousLoop())
    orch._memory_store["final_report"] = ResearchReport("current query", "current draft")
    orch._config = RunConfig(adversarial_timeout_seconds=0)
    orch._start_time = time.monotonic()
    asyncio.run(orch._do_adversarial())
    report = orch._memory_store["final_report"]
    assert report.query == "current query"
    assert report.content == "current draft"
    assert report.adversarial_history == []
