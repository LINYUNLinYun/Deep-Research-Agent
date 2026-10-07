#!/usr/bin/env python3
"""Paired Critic-Repairer ablation over previously generated ResearchBench reports."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import ResearchBench
from src.adversarial.repairer_agent import RepairerAgent
from src.adversarial.loop import AdversarialLoop
from src.adversarial.critic_agent import CriticAgent
from src.core.runner import _create_tools_factory, load_config
from src.models.model_router import ModelRouter
from src.orchestrator.schemas import ResearchReport


DEFAULT_QUESTION_IDS = [
    "tech_005",
    "law_001",
    "tech_003",
    "fin_005",
    "med_005",
    "cross_001",
]

BASELINE_EXPERIMENTS = (
    "baseline-no-numeric-miner-v1",
    "baseline-no-numeric-development-v1",
)


class CountingPolicy:
    """Transparent policy proxy used to retain call counts per question."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls = 0

    @property
    def max_tokens(self) -> int:
        return self.inner.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.inner.max_tokens = value

    def __call__(self, messages: list[dict[str, Any]]) -> Any:
        self.calls += 1
        return self.inner(messages)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def locate_baseline_reports(root: Path) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for experiment in BASELINE_EXPERIMENTS:
        for path in (root / experiment / "reports").glob("*.md"):
            reports[path.stem] = path.resolve()
    return reports


def extract_sources(report_text: str) -> list[dict[str, str]]:
    """Recover structured sources from the report's generated appendix."""
    parts = re.split(
        r"(?im)^\s{0,3}#{1,6}\s*(?:参考来源|参考文献|来源列表|references?|sources?)\s*:?[ \t]*$",
        report_text,
        maxsplit=1,
    )
    if len(parts) != 2:
        return []
    chunks = re.split(r"(?m)(?=^\[\d+\]\s+\[)", parts[1])
    sources: list[dict[str, str]] = []
    for chunk in chunks:
        match = re.match(
            r"(?s)^\[(\d+)\]\s+\[(.*?)\]\((https?://.*?)\)\s+[—-]\s+(.*)$",
            chunk.strip(),
        )
        if not match:
            continue
        _, title, url, snippet = match.groups()
        sources.append({
            "title": " ".join(title.split()),
            "url": url.strip(),
            "snippet": " ".join(snippet.split())[:1200],
        })
    return sources


def metric_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    keys = sorted(set(baseline.get("metrics", {})) | set(candidate.get("metrics", {})))
    deltas = {
        key: round(
            float(candidate.get("metrics", {}).get(key, 0.0))
            - float(baseline.get("metrics", {}).get(key, 0.0)),
            6,
        )
        for key in keys
    }
    deltas["composite_score"] = round(
        float(candidate.get("composite_score", 0.0))
        - float(baseline.get("composite_score", 0.0)),
        6,
    )
    deltas["hallucination_rate"] = round(
        float(candidate.get("hallucination_rate", 0.0))
        - float(baseline.get("hallucination_rate", 0.0)),
        6,
    )
    return deltas


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in details if item.get("status") == "completed"]
    deltas = [item["deltas"]["composite_score"] for item in completed]
    tolerance = 1e-6
    adversarial_statuses: dict[str, int] = {}
    for item in completed:
        status = str(item.get("adversarial_status", "unknown"))
        adversarial_statuses[status] = adversarial_statuses.get(status, 0) + 1
    modified_reports = 0
    for item in completed:
        try:
            baseline = Path(item["baseline_path"]).read_text(encoding="utf-8")
            candidate = Path(item["critic_repairer_report_path"]).read_text(encoding="utf-8")
            modified_reports += baseline != candidate
        except (KeyError, OSError):
            continue
    return {
        "requested": len(details),
        "completed": len(completed),
        "failed": len(details) - len(completed),
        "baseline_average": round(
            sum(item["baseline_evaluation"]["composite_score"] for item in completed)
            / max(len(completed), 1),
            6,
        ),
        "critic_repairer_average": round(
            sum(item["critic_repairer_evaluation"]["composite_score"] for item in completed)
            / max(len(completed), 1),
            6,
        ),
        "average_delta": round(sum(deltas) / max(len(deltas), 1), 6),
        "wins": sum(delta > tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta < -tolerance for delta in deltas),
        "modified_reports": modified_reports,
        "adversarial_statuses": adversarial_statuses,
        "critic_calls": sum(int(item.get("critic_calls", 0)) for item in details),
        "repairer_calls": sum(int(item.get("repairer_calls", 0)) for item in details),
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds", 0.0)) for item in details), 3),
    }


async def run_question(
    *,
    question: dict[str, Any],
    baseline_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    bench: ResearchBench,
) -> dict[str, Any]:
    question_id = str(question["id"])
    baseline_text = baseline_path.read_text(encoding="utf-8")
    sources = extract_sources(baseline_text)
    if not sources:
        raise ValueError(f"no structured sources recovered for {question_id}")

    model_cfg = config["model"]
    sampling = model_cfg["backend_sampling"]
    adversarial_cfg = config["adversarial"]
    backend = model_cfg["backend_mapping"]["critic_agent"]

    def policy(module: str) -> CountingPolicy:
        kwargs = dict(sampling.get(backend, {}))
        kwargs.update(sampling.get("modules", {}).get(module, {}))
        kwargs["use_cache"] = False
        return CountingPolicy(ModelRouter.create_backend(
            backend,
            module_name=f"{module}_ablation_{question_id}",
            **kwargs,
        ))

    critic_policy = policy("critic_agent")
    repairer_policy = policy("repairer_agent")
    tools = _create_tools_factory(copy.deepcopy(config))
    max_issues = int(adversarial_cfg.get("max_issues_per_round", 5))
    critic = CriticAgent(
        critic_policy,
        max_tokens=int(adversarial_cfg.get("critic_max_tokens", 4096)),
        max_issues=max_issues,
        max_issues_per_dimension=int(adversarial_cfg.get("max_issues_per_dimension", 2)),
        max_sources=int(adversarial_cfg.get("max_sources", 20)),
        context_chars=int(adversarial_cfg.get("critic_context_chars", 8000)),
        dimension_parse_retries=int(adversarial_cfg.get("critic_dimension_parse_retries", 1)),
    )
    repairer = RepairerAgent(
        repairer_policy,
        tools=tools,
        max_tokens=int(adversarial_cfg.get("repairer_max_tokens", 4096)),
        max_issues=max_issues,
        max_sources=int(adversarial_cfg.get("max_sources", 20)),
        max_consecutive_failures=int(adversarial_cfg.get("max_consecutive_failures", 2)),
        max_candidate_sources=int(adversarial_cfg.get("max_candidate_sources", 3)),
        repair_context_chars=int(adversarial_cfg.get("repairer_context_chars", 4000)),
        self_verify_context_chars=int(adversarial_cfg.get("self_verify_context_chars", 6000)),
    )
    loop = AdversarialLoop(
        critic,
        repairer,
        max_rounds=int(adversarial_cfg.get("max_rounds", 3)),
        score_threshold=float(adversarial_cfg.get("score_threshold", 8.0)),
        delta_threshold=float(adversarial_cfg.get("delta_threshold", 0.3)),
        evidence_verifier=None,
    )
    report = ResearchReport(
        query=str(question["query"]),
        content=baseline_text,
        sources=sources,
        confidence=0.0,
    )
    baseline_evaluation = bench.evaluate_report(baseline_text, question_id)
    started = time.monotonic()
    timeout = float(adversarial_cfg.get("timeout_seconds", 180))
    try:
        result, history = await asyncio.wait_for(loop.run(report), timeout=timeout)
        run_status = "completed"
        error = ""
    except asyncio.TimeoutError:
        result, history = report, []
        result.adversarial_status = "skipped"
        result.adversarial_reason = "adversarial_timeout"
        run_status = "failed"
        error = "adversarial_timeout"
    elapsed = time.monotonic() - started

    candidate_path = output_dir / "reports" / f"{question_id}.md"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(result.content, encoding="utf-8")
    history_path = output_dir / "history" / f"{question_id}.json"
    atomic_json(history_path, {"question_id": question_id, "history": history})
    critic_repairer_evaluation = bench.evaluate_report(result.content, question_id)
    return {
        "question_id": question_id,
        "domain": question.get("domain", ""),
        "status": run_status,
        "error": error,
        "baseline_path": str(baseline_path),
        "critic_repairer_report_path": str(candidate_path.resolve()),
        "history_path": str(history_path.resolve()),
        "sources_count": len(sources),
        "elapsed_seconds": round(elapsed, 3),
        "critic_calls": critic_policy.calls,
        "repairer_calls": repairer_policy.calls,
        "adversarial_status": result.adversarial_status,
        "adversarial_reason": result.adversarial_reason,
        "adversarial_rounds": result.adversarial_rounds,
        "baseline_evaluation": baseline_evaluation,
        "critic_repairer_evaluation": critic_repairer_evaluation,
        "deltas": metric_deltas(baseline_evaluation, critic_repairer_evaluation),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    output_dir = Path(args.output_dir).resolve() / args.experiment_id
    result_path = output_dir / "results.json"
    question_ids = args.question_ids or DEFAULT_QUESTION_IDS
    reports = locate_baseline_reports(Path(args.baseline_root).resolve())
    bench = ResearchBench()
    questions = {str(item["id"]): item for item in bench.questions}
    missing = [qid for qid in question_ids if qid not in reports or qid not in questions]
    if missing:
        raise ValueError(f"missing baseline report or benchmark question: {missing}")

    payload: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "question_ids": question_ids,
        "backend": config["model"]["backend_mapping"]["critic_agent"],
        "adversarial_config": copy.deepcopy(config["adversarial"]),
        "details": [],
    }
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"experiment exists; pass --resume: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("question_ids") != question_ids:
            raise ValueError("resume question_ids mismatch")
        if payload.get("backend") != config["model"]["backend_mapping"]["critic_agent"]:
            raise ValueError("resume backend mismatch")

    completed = {item["question_id"] for item in payload.get("details", [])}
    for index, question_id in enumerate(question_ids, 1):
        if question_id in completed:
            print(f"[{index}/{len(question_ids)}] SKIP {question_id} (checkpointed)", flush=True)
            continue
        print(f"[{index}/{len(question_ids)}] START {question_id}", flush=True)
        ModelRouter.clear_cache()
        try:
            detail = await run_question(
                question=questions[question_id],
                baseline_path=reports[question_id],
                config=config,
                output_dir=output_dir,
                bench=bench,
            )
        except Exception as exc:
            detail = {
                "question_id": question_id,
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": 0.0,
                "critic_calls": 0,
                "repairer_calls": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            ModelRouter.clear_cache()
            try:
                from src.tools.web_search import WebSearchTool
                await WebSearchTool.close_session()
            except Exception:
                pass
        payload.setdefault("details", []).append(detail)
        payload["summary"] = summarize(payload["details"])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_path, payload)
        print(
            f"[{index}/{len(question_ids)}] END {question_id} "
            f"status={detail.get('adversarial_status', detail.get('status'))} "
            f"delta={detail.get('deltas', {}).get('composite_score')} "
            f"calls={detail.get('critic_calls', 0)}+{detail.get('repairer_calls', 0)}",
            flush=True,
        )
    payload["summary"] = summarize(payload.get("details", []))
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(result_path, payload)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline-root", default="outputs/evaluation")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--experiment-id", default="critic_repairer-baseline-stage1-v1")
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    path = asyncio.run(run(parse_args()))
    print(f"Results: {path}")
