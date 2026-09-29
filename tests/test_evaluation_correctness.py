"""Regression tests for evaluation correctness and reproducibility."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from evaluation.benchmarks.hotpotqa import HotpotQABenchmark
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.metrics.rule_based import RuleBasedMetrics
from evaluation.metrics.composite import compute_composite_score
from evaluation.metrics.stats import bootstrap_ci_paired
from src.core.ablation import AblationStudy
from src.harness_evolution.experiment import _ensure_policy_claim_verification
from scripts.run_eval import load_manifest_questions
from src.agents.summarizer import SummarizerAgent
from src.orchestrator.schemas import AgentResult, AgentStatus


def test_researchbench_domain_aliases_match_builtin_labels() -> None:
    bench = ResearchBench()
    assert len(bench.get_questions(domain="tech")) > 0
    assert len(bench.get_questions(domain="med")) > 0
    assert len(bench.get_questions(domain="fin")) > 0
    assert len(bench.get_questions(domain="科技")) == len(bench.get_questions(domain="technology"))


def test_factual_aliases_are_normalized_in_composite() -> None:
    score = RuleBasedMetrics.composite_score(
        {
            "factual_accuracy_str": 1.0,
            "factual_accuracy_sem": 1.0,
            "logical_consistency": 1.0,
            "citation_coverage": 1.0,
            "bias": 1.0,
            "comprehensiveness": 1.0,
        }
    )
    assert score == 1.0
    assert compute_composite_score(
        {"factual_accuracy_str": 1.0, "factual_accuracy_sem": 1.0}
    )["rule_score"] == 1.0


def test_false_supported_rate_uses_verified_contradictions_and_fails_closed() -> None:
    assert RuleBasedMetrics.false_supported_rate({
        "total_claims": 4,
        "supported": 2,
        "contradicted": 1,
        "unknown": 1,
    }) == 0.25
    assert RuleBasedMetrics.false_supported_rate({"total_claims": 0, "contradicted": 0}) is None
    assert RuleBasedMetrics.false_supported_rate({"error": "verification failed"}) is None


def test_policy_evaluation_runs_claim_verifier_when_main_path_skipped_it() -> None:
    report = SimpleNamespace(evidence_verification={})

    class Verifier:
        async def verify(self, observed_report):
            assert observed_report is report
            return ["verified"]

        @staticmethod
        def summary(results):
            assert results == ["verified"]
            return {"total_claims": 2, "supported": 1, "contradicted": 1, "unknown": 0}

    summary = asyncio.run(_ensure_policy_claim_verification({
        "last_report": report,
        "evidence_verifier": Verifier(),
    }))

    assert summary["contradicted"] == 1
    assert report.evidence_verification == summary


def test_citation_coverage_excludes_reference_appendix() -> None:
    report = (
        "结论：模型性能提升。[1]\n\n"
        "## 参考来源\n"
        "- [1] https://example.com/source\n"
        "- [2] https://example.com/another"
    )
    assert RuleBasedMetrics.citation_coverage(report) == 1.0


def test_citation_coverage_rejects_unknown_reference_id() -> None:
    report = "结论：模型性能提升。[99]\n\n## 参考来源\n[1] https://example.com/source"
    assert RuleBasedMetrics.citation_coverage(report) == 0.0


def test_citation_coverage_ignores_runtime_metadata() -> None:
    report = (
        "结论：模型性能提升。[1]\n\n"
        "## 元信息\n\n- **搜索轮数**: 2\n\n"
        "## 参考来源\n\n[1] https://example.com/source"
    )
    assert RuleBasedMetrics.citation_coverage(report) == 1.0


def test_summarizer_builds_bounded_numbered_source_catalog() -> None:
    result = AgentResult(
        "task_1",
        AgentStatus.SUCCESS,
        output="finding",
        confidence=0.8,
        trajectory=[{
            "role": "tool",
            "result": {
                "results": [{
                    "title": "Official report",
                    "url": "https://example.com/report",
                    "snippet": "Revenue reached 20% in 2024.",
                }]
            },
        }],
    )
    summarizer = SummarizerAgent("s", policy=lambda _: {"content": ""})
    catalog = summarizer._collect_sources("revenue", [result])
    prompt = summarizer._build_synthesis_prompt("revenue", [result], catalog)
    assert catalog[0]["citation_id"] == 1
    assert "[1] Official report" in prompt
    assert "using [N]" in prompt
    assert "refer-then-claim" in prompt
    assert catalog[0]["source_id"].startswith("src_")


def test_summarizer_catalog_obeys_count_and_character_budgets() -> None:
    results = []
    for index in range(10):
        results.append(AgentResult(
            f"task_{index}", AgentStatus.SUCCESS, output="finding", confidence=0.8,
            trajectory=[{"role": "tool", "result": {"results": [{
                "title": f"Source {index}",
                "url": f"https://example{index}.com/report",
                "snippet": "evidence " * 200,
            }]}}],
        ))
    summarizer = SummarizerAgent(
        "s", policy=lambda _: {"content": ""},
        max_catalog_sources=3, max_catalog_chars=1800,
    )
    catalog = summarizer.collect_sources("q", results)
    prompt = summarizer._build_synthesis_prompt("q", results, catalog)
    assert len(catalog) <= 3
    assert all(len(source["source_span"]) <= 500 for source in catalog)
    assert len(prompt.split("# Source Catalog", 1)[1]) < 3000


def test_summarizer_marks_current_source_for_latest_query() -> None:
    current_year = datetime.now().astimezone().year
    assert SummarizerAgent._temporal_relevance(
        "latest model progress", f"{current_year}-03-01"
    ) == "current"


def test_evidence_ledger_preserves_browser_source_and_full_span() -> None:
    result = AgentResult(
        "task_browser",
        AgentStatus.SUCCESS,
        output="finding",
        trajectory=[{
            "role": "tool",
            "name": "browser",
            "args": {"url": "https://example.com/article?utm_source=test"},
            "result": "Full article passage with the exact supporting fact.",
        }],
    )
    catalog = SummarizerAgent("s", policy=lambda _: {"content": ""})._collect_sources("q", [result])
    assert catalog[0]["url"] == "https://example.com/article"
    assert catalog[0]["source_span"].startswith("Full article passage")
    assert catalog[0]["content_hash"]
    assert catalog[0]["source_cluster_id"] == "domain:example.com"


def test_hotpot_answer_extraction_skips_markdown_title() -> None:
    report = "# 研究报告：曹雪芹\n\n## 答案\n清朝。[1]\n\n## 参考来源\n- [1] https://example.com"
    assert HotpotQABenchmark.extract_answer(report) == "清朝。"


def test_paired_bootstrap_is_seeded_and_uses_null_sign_flip() -> None:
    first = bootstrap_ci_paired([0.2, 0.3, 0.4], n_bootstrap=200, seed=7)
    second = bootstrap_ci_paired([0.2, 0.3, 0.4], n_bootstrap=200, seed=7)
    assert first == second
    assert first["mean_diff"] == 0.3
    assert 0.0 <= first["p_value"] <= 1.0
    # With only three paired observations the exact sign-flip test has a
    # coarse 1/9 resolution (the all-positive result is p=2/9 with the
    # conservative +1 correction).
    assert first["p_value"] <= 0.25


def test_no_compressor_ablation_disables_both_switches() -> None:
    _, overrides = AblationStudy.DEFAULT_MODULE_ABLATIONS["no_compressor"]
    assert overrides["compressor"]["enabled"] is False
    assert overrides["compressor"]["enable_multilevel"] is False


def test_manifest_question_selection_respects_split_and_limit() -> None:
    questions, manifest = load_manifest_questions(
        "configs/harness_evolution/datasets/researchbench_v1.yaml",
        "miner",
        2,
    )
    assert [item["id"] for item in questions] == ["tech_001", "med_001"]
    assert manifest.data["version"] == "v1"
