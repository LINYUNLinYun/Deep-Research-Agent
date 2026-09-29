#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_eval.py
================================================================================
标准评测集入口脚本（合并了原 run_evaluation.py）。

支持:
  --benchmark research_bench : 自建深度研究评测集（规则指标）
  --benchmark hotpotqa      : 公共多跳 QA 评测集（EM/F1）

Usage:
    python scripts/run_eval.py --benchmark research_bench --num_questions 20
    python scripts/run_eval.py --benchmark hotpotqa --num_questions 100
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
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

from src.core.runner import collect_harness_telemetry, initialize_modules, load_config, run_research, setup_logging
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.benchmarks.hotpotqa import HotpotQABenchmark
from evaluation.report import EvaluationReport
from src.harness_evolution.evaluation import DatasetManifest
from src.harness_evolution.manifest import RunManifest, stable_hash
from src.harness_evolution.registry import VersionRegistry
from src.models.model_router import ModelRouter


ARTIFACT_IDS = ("search_control_policy", "verify_numeric_claim_skill")


def load_manifest_questions(
    manifest_path: str | Path,
    split: str,
    num_questions: int,
) -> tuple[list[dict[str, Any]], DatasetManifest]:
    """Resolve an explicit, versioned ResearchBench split in manifest order."""
    manifest = DatasetManifest.load(manifest_path)
    splits = manifest.data.get("splits", {})
    if split not in splits:
        raise ValueError(f"dataset split does not exist: {split}")
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")

    by_id = {str(item["id"]): item for item in ResearchBench.DEFAULT_QUESTIONS}
    selected: list[dict[str, Any]] = []
    for question_id in splits[split][:num_questions]:
        if question_id not in by_id:
            raise ValueError(f"dataset manifest references unknown question: {question_id}")
        selected.append(copy.deepcopy(by_id[question_id]))
    if not selected:
        raise ValueError(f"dataset split is empty: {split}")
    return selected, manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON checkpoint without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _registry_path(registry_dir: Path, registry: str) -> Path:
    path = Path(registry)
    if path.suffix not in {".yaml", ".yml"}:
        path = path.with_suffix(".yaml")
    if not path.is_absolute():
        path = registry_dir / path
    return path.resolve()


def _manifest_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in details if item.get("status") == "success"]
    scores = [float(item.get("composite_score", 0.0)) for item in successes]
    return {
        "average_composite": sum(scores) / len(scores) if scores else 0.0,
        "num_success": len(successes),
        "num_failed": len(details) - len(successes),
        "completed_question_ids": [item.get("question_id") for item in details],
    }


def evaluate_manifest_split(
    *,
    manifest_path: str | Path,
    split: str,
    num_questions: int,
    config: dict[str, Any],
    registry: str,
    experiment_id: str,
    output_dir: str | Path,
    resume: bool,
) -> tuple[dict[str, Any], Path]:
    """Run an isolated, incrementally checkpointed ResearchBench experiment."""
    logger = logging.getLogger("run_eval")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id):
        raise ValueError("experiment_id contains unsupported characters")

    questions, dataset_manifest = load_manifest_questions(manifest_path, split, num_questions)
    cfg = copy.deepcopy(config)
    harness_cfg = cfg.setdefault("harness_evolution", {})
    harness_cfg["enabled"] = True
    harness_cfg["registry"] = registry

    registry_dir = Path(str(harness_cfg.get("registry_dir", "configs/harness_evolution/registries")))
    if not registry_dir.is_absolute():
        registry_dir = PROJECT_ROOT / registry_dir
    manager = VersionRegistry(registry_dir)
    artifacts = {artifact_id: manager.resolve(registry, artifact_id) for artifact_id in ARTIFACT_IDS}
    resolved_registry = _registry_path(registry_dir, registry)

    experiment_dir = Path(output_dir).resolve() / experiment_id
    reports_dir = experiment_dir / "reports"
    manifests_dir = experiment_dir / "manifests"
    telemetry_dir = experiment_dir / "telemetry"
    result_path = experiment_dir / "results.json"
    config_sha256 = stable_hash(cfg)
    identity = {
        "experiment_id": experiment_id,
        "dataset_sha256": dataset_manifest.sha256,
        "dataset_version": dataset_manifest.data.get("version", ""),
        "split": split,
        "registry": str(resolved_registry),
        "config_sha256": config_sha256,
    }
    payload: dict[str, Any] = {**identity, "requested_question_ids": [q["id"] for q in questions], "details": []}

    if result_path.exists():
        if not resume:
            raise FileExistsError(f"experiment already exists; pass --resume: {result_path}")
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        for key, expected in identity.items():
            if existing.get(key) != expected:
                raise ValueError(f"resume identity mismatch for {key}")
        completed = {str(item.get("question_id")) for item in existing.get("details", [])}
        requested = {str(item["id"]) for item in questions}
        if not completed <= requested:
            raise ValueError("resume request omits previously completed questions")
        payload = existing
        payload["requested_question_ids"] = [q["id"] for q in questions]

    completed_ids = {str(item.get("question_id")) for item in payload["details"]}
    bench = ResearchBench()
    total = len(questions)
    for index, question in enumerate(questions, 1):
        question_id = str(question["id"])
        if question_id in completed_ids:
            logger.info("[%s/%s] 跳过已完成题目: %s", index, total, question_id)
            continue

        run_id = f"{experiment_id}-{question_id}-{uuid.uuid4().hex[:12]}"
        session_id = f"eval:{run_id}"
        created_at = datetime.now(timezone.utc).isoformat()
        run_manifest = RunManifest.create(
            run_id=run_id,
            experiment_id=experiment_id,
            pair_id=f"{question_id}:r1",
            query_id=question_id,
            group="production_baseline" if registry == "production" else registry,
            registry=resolved_registry,
            artifacts=artifacts,
            config=cfg,
            executor={
                "backend": cfg.get("model", {}).get("backend", ""),
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                "sampling": cfg.get("model", {}).get("backend_sampling", {}),
            },
            budgets={
                "timeout_seconds": cfg.get("orchestrator", {}).get("global_timeout_seconds"),
                "token_budget": cfg.get("orchestrator", {}).get("token_budget"),
                "max_replan_rounds": cfg.get("orchestrator", {}).get("max_replan_rounds"),
                "max_tool_calls_per_subagent": cfg.get("planner", {}).get("max_tool_calls_per_subagent"),
            },
            memory_session=session_id,
            dataset={
                "id": dataset_manifest.data.get("dataset_id", "researchbench"),
                "version": dataset_manifest.data.get("version", ""),
                "split": split,
                "sha256": dataset_manifest.sha256,
                "manifest_path": str(Path(manifest_path).resolve()),
            },
            tool={"mode": "live", "fixture_path": ""},
            evaluator={"id": "researchbench_rules", "version": "v1"},
            created_at=created_at,
            repo_root=PROJECT_ROOT,
            metadata={
                "adversarial_enabled": bool(cfg.get("adversarial", {}).get("enabled", True)),
                "numeric_verification_enabled": bool(
                    cfg.get("adversarial", {}).get("evidence_verification_enabled", False)
                ),
            },
        )
        manifest_path_for_run = manifests_dir / f"{run_id}.json"
        manifest_sha256 = run_manifest.write(manifest_path_for_run)
        logger.info("[%s/%s] 开始题目 %s (run_id=%s)", index, total, question_id, run_id)

        started = time.time()
        modules: dict[str, Any] = {}
        try:
            ModelRouter.clear_cache()
            modules = initialize_modules(cfg, session_id=session_id)
            report_text = asyncio.run(run_research(str(question["query"]), cfg, modules))
            elapsed = time.time() - started
            evaluation = bench.evaluate_report(report_text, question_id)
            telemetry = collect_harness_telemetry(modules)
            report_file = reports_dir / f"{question_id}.md"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(report_text, encoding="utf-8")
            telemetry_file = telemetry_dir / f"{question_id}.json"
            _atomic_json(telemetry_file, telemetry)
            detail = {
                **evaluation,
                "status": "success",
                "question_id": question_id,
                "run_id": run_id,
                "elapsed_seconds": elapsed,
                "manifest_path": str(manifest_path_for_run),
                "manifest_sha256": manifest_sha256,
                "report_path": str(report_file),
                "telemetry_path": str(telemetry_file),
                "harness": telemetry,
            }
            logger.info("  → composite=%.3f, time=%.1fs", detail["composite_score"], elapsed)
        except Exception as exc:
            elapsed = time.time() - started
            logger.exception("  → FAILED: %s", exc)
            detail = {
                "status": "failed",
                "question_id": question_id,
                "run_id": run_id,
                "error": str(exc),
                "composite_score": 0.0,
                "elapsed_seconds": elapsed,
                "manifest_path": str(manifest_path_for_run),
                "manifest_sha256": manifest_sha256,
            }
            try:
                from src.tools.web_search import WebSearchTool
                asyncio.run(WebSearchTool.close_session())
            except Exception:
                logger.debug("failed to close web session after evaluation error", exc_info=True)
        finally:
            ModelRouter.clear_cache()

        payload["details"].append(detail)
        payload["summary"] = _manifest_summary(payload["details"])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(result_path, payload)

    payload["summary"] = _manifest_summary(payload["details"])
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(result_path, payload)
    return payload, result_path


def evaluate_research_bench(
    num_questions: int,
    domain: str | None,
    config: dict,
) -> EvaluationReport:
    """在 ResearchBench 上运行评测。"""
    logger = logging.getLogger("run_eval")
    bench = ResearchBench()
    questions = bench.get_questions(domain=domain, n=num_questions)
    logger.info(f"ResearchBench 加载 {len(questions)} 道题目")

    modules = initialize_modules(config)
    base_session = getattr(modules.get("memory_store"), "session_id", "")
    report = EvaluationReport(name="ResearchBench_Evaluation", num_questions=len(questions))

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["query"]
        logger.info(f"[{idx}/{len(questions)}] 评测题目: {qid}")

        start = time.time()
        try:
            if modules.get("memory_store") is not None:
                modules["memory_store"].set_session(f"{base_session}:{qid}")
            report_text = asyncio.run(run_research(query, config, modules))
            elapsed = time.time() - start

            eval_result = bench.evaluate_report(report_text, qid)
            eval_result["elapsed_seconds"] = elapsed
            eval_result["harness"] = collect_harness_telemetry(modules)
            report.add_detail(eval_result)
            logger.info(f"  → composite={eval_result['composite_score']:.3f}, time={elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"  → FAILED: {e}")
            report.add_detail({
                "question_id": qid,
                "error": str(e),
                "composite_score": 0.0,
            })

    # 汇总
    valid_scores = [d["composite_score"] for d in report.details if "composite_score" in d]
    report.set_summary({
        "average_composite": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "num_success": len([d for d in report.details if "error" not in d]),
        "num_failed": len([d for d in report.details if "error" in d]),
    })

    return report


def evaluate_hotpotqa(
    num_questions: int,
    config: dict,
    use_mock: bool = False,
) -> EvaluationReport:
    """在 HotpotQA 上运行评测（深度研究变体：评估完整报告质量）。"""
    logger = logging.getLogger("run_eval")
    bench = HotpotQABenchmark(use_mock=use_mock)
    questions = bench.get_samples(n=num_questions, shuffle=True)
    logger.info(f"HotpotQA 加载 {len(questions)} 道题目")

    modules = initialize_modules(config)
    base_session = getattr(modules.get("memory_store"), "session_id", "")
    report = EvaluationReport(name="HotpotQA_DeepResearch_Evaluation", num_questions=len(questions))

    predictions = []
    for idx, q in enumerate(questions, 1):
        query = q["query"]
        gold = q["expected_answer"]
        logger.info(f"[{idx}/{len(questions)}] 评测: {query[:60]}...")

        try:
            if modules.get("memory_store") is not None:
                modules["memory_store"].set_session(f"{base_session}:hotpot:{idx}")
            report_text = asyncio.run(run_research(query, config, modules))
            # Reports normally start with a Markdown title; extracting the
            # first line made HotpotQA EM/F1 score the title instead of the
            # answer.  Prefer an explicit answer/conclusion section.
            pred_answer = HotpotQABenchmark.extract_answer(report_text)
        except Exception as e:
            logger.warning(f"  → FAILED: {e}")
            pred_answer = ""
            report_text = ""

        predictions.append({
            "query_id": idx,
            "prediction": pred_answer,
            "gold": gold,
            "report": report_text,
        })

        depth = bench.evaluate_report(report_text, gold) if report_text else {}
        report.add_detail({
            "query_id": idx,
            "query": query,
            "prediction": pred_answer,
            "gold": gold,
            "depth_metrics": depth,
            "harness": collect_harness_telemetry(modules) if report_text else {},
        })

    metrics = bench.evaluate(predictions, metrics=["em", "f1", "pass@1"])
    report.set_summary(metrics)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch Agent 标准评测脚本")
    parser.add_argument("--benchmark", type=str, choices=["research_bench", "hotpotqa"],
                        required=True, help="评测基准")
    parser.add_argument("--num_questions", type=int, default=20, help="评测题目数量")
    parser.add_argument("--domain", type=str, default=None, help="领域过滤（仅 ResearchBench）")
    parser.add_argument("--use_mock", action="store_true", help="使用内置 mock 数据（仅 HotpotQA，用于流程验证）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="输出目录")
    parser.add_argument("--dataset_manifest", type=str, default=None,
                        help="版本化 ResearchBench split manifest")
    parser.add_argument("--split", type=str, choices=["miner", "development", "held_out"], default="miner")
    parser.add_argument("--registry", type=str, default="production",
                        help="显式 Harness registry 名称或 YAML 路径")
    parser.add_argument("--experiment_id", type=str, default=None,
                        help="可恢复评测的稳定实验 ID")
    parser.add_argument("--resume", action="store_true", help="从同一实验结果继续未完成题目")
    parser.add_argument(
        "--no-numeric-verification",
        action="store_true",
        help="关闭 claim-level Numeric Verification（用于无 Numeric 的隔离 baseline）",
    )
    parser.add_argument(
        "--research-state-shadow",
        action="store_true",
        help="启用策略二 coverage/frontier graph 的 shadow trace，不改变 DAG 调度",
    )
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    config = load_config(args.config)
    if args.no_numeric_verification:
        config.setdefault("adversarial", {})["evidence_verification_enabled"] = False
    if args.research_state_shadow:
        state_cfg = config.setdefault("planner", {}).setdefault("research_state", {})
        state_cfg["enabled"] = True
        state_cfg["active"] = False
    logger.info(f"配置加载完成: {args.config or 'configs/default.yaml'}")

    if args.benchmark == "research_bench" and args.dataset_manifest:
        experiment_id = args.experiment_id or f"researchbench-{args.split}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        payload, filepath = evaluate_manifest_split(
            manifest_path=args.dataset_manifest,
            split=args.split,
            num_questions=args.num_questions,
            config=config,
            registry=args.registry,
            experiment_id=experiment_id,
            output_dir=args.output_dir,
            resume=args.resume,
        )
        logger.info("增量评测结果已保存: %s", filepath)
        print("\n" + "=" * 60)
        print("评测摘要")
        print("=" * 60)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print("=" * 60)
        return
    if args.benchmark == "research_bench":
        report = evaluate_research_bench(args.num_questions, args.domain, config)
    elif args.benchmark == "hotpotqa":
        report = evaluate_hotpotqa(args.num_questions, config, use_mock=args.use_mock)
    else:
        raise ValueError(f"未知基准: {args.benchmark}")

    filepath = report.save(args.output_dir)
    logger.info(f"评测报告已保存: {filepath}")

    print("\n" + "=" * 60)
    print("评测摘要")
    print("=" * 60)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
