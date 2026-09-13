#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/ablation.py
================================================================================
消融实验通用框架。

对外接口:
    - AblationStudy.run_module_ablation(config, questions, systems) -> dict
    - AblationStudy.run_rounds_ablation(config, questions, max_rounds) -> dict
================================================================================
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable

from .runner import collect_harness_telemetry, initialize_modules, run_research

logger = logging.getLogger("ablation")


class AblationStudy:
    """消融实验框架：支持模块消融和对抗轮数消融。"""

    # 模块消融的默认配置映射
    DEFAULT_MODULE_ABLATIONS: dict[str, tuple[str, dict]] = {
        "full": ("完整系统", {}),
        "no_adversarial": ("关闭对抗降噪", {"adversarial": {"enabled": False}}),
        # Disable both the high-level switch and legacy multilevel flag.  The
        # runner now honours ``compressor.enabled`` while older configs only
        # inspected ``enable_multilevel``; setting both makes the ablation
        # semantically unambiguous across versions.
        "no_compressor": (
            "关闭上下文压缩",
            {"compressor": {"enabled": False, "enable_multilevel": False}},
        ),
        "no_memory": ("关闭记忆存储", {"memory": {"enabled": False}}),
    }

    HARNESS_ABLATIONS: dict[str, tuple[str, dict]] = {
        "harness_full": ("完整 Harness", {}),
        "no_search_control": (
            "关闭 query rewrite 和跨 Agent 搜索去重",
            {"tools": {"search_control": {"enabled": False}}},
        ),
        "no_evidence_verifier": (
            "关闭 claim-level evidence verification",
            {"adversarial": {"evidence_verification_enabled": False}},
        ),
        "static_replan": (
            "关闭状态驱动 replan",
            {"planner": {"enable_replan": False}},
        ),
        "no_harness_evolution": (
            "关闭版本化 Search Policy 与 Numeric Skill",
            {"harness_evolution": {"enabled": False}},
        ),
    }

    @staticmethod
    def override_config(config: dict, overrides: dict) -> dict:
        """深度合并配置覆盖（支持嵌套字典）。"""
        cfg = copy.deepcopy(config)

        def _deep_merge(base: dict, patch: dict) -> dict:
            for key, value in patch.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = _deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        return _deep_merge(cfg, overrides)

    @staticmethod
    def _score_report(
        report: str,
        question: dict[str, Any],
        evaluator: Callable[[str, dict[str, Any]], float | dict[str, Any]] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Score a report without fabricating a ``1.0`` success score.

        Callers can inject a project-specific evaluator.  For ResearchBench
        shaped questions we provide a lightweight built-in evaluator so the
        generic ablation API remains useful, while preserving a deterministic
        success fallback for arbitrary question dictionaries that contain no
        expected facts/topics.
        """
        if evaluator is not None:
            value = evaluator(report, question)
            if isinstance(value, dict):
                score = float(value.get("composite_score", value.get("score", 0.0)))
                return score, value
            return float(value), {"composite_score": float(value)}

        if question.get("ground_truth") or question.get("expected_topics"):
            # Import lazily to keep core usable without the optional evaluation
            # package at module import time.
            from evaluation.metrics.rule_based import RuleBasedMetrics

            ground_truth = question.get("ground_truth", {})
            expected_topics = question.get("expected_topics", [])
            factual = RuleBasedMetrics.fact_accuracy(report, ground_truth)
            metrics = {
                "factual_accuracy": factual,
                "logical_consistency": RuleBasedMetrics.logical_consistency(report),
                "citation_coverage": RuleBasedMetrics.citation_coverage(report),
                "bias": max(0.0, 1.0 - RuleBasedMetrics.hallucination_rate(report)),
                "comprehensiveness": RuleBasedMetrics.comprehensiveness(report, expected_topics),
            }
            composite = RuleBasedMetrics.composite_score(metrics)
            return composite, {"composite_score": composite, "metrics": metrics}

        # No oracle was supplied; report success is a status signal, not a
        # quality score.  Keep the old shape but label it explicitly.
        return (1.0 if report else 0.0), {"status_score": 1.0 if report else 0.0}

    # -----------------------------------------------------------------------
    # 模块消融：full / no_XXX
    # -----------------------------------------------------------------------
    @classmethod
    def run_module_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        systems: dict[str, tuple[str, dict]] | None = None,
        evaluator: Callable[[str, dict[str, Any]], float | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        运行模块消融实验。

        Args:
            config: 基础配置。
            questions: 评测题目列表（每条含 id, query）。
            systems: 消融配置映射。键为 system_name，值为 (描述, 配置覆盖)。
                     默认使用 DEFAULT_MODULE_ABLATIONS。

        Returns:
            包含各系统得分和明细的字典。
        """
        if systems is None:
            systems = cls.DEFAULT_MODULE_ABLATIONS

        results: list[dict[str, Any]] = []

        for name, (desc, overrides) in systems.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"[消融实验] {name}: {desc}")
            logger.info(f"{'='*60}")

            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)
            base_session = getattr(modules.get("memory_store"), "session_id", "")

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for q in questions:
                qid = q.get("id", "unknown")
                query = q.get("query", "")
                logger.info(f"  [{qid}] {query[:60]}...")

                start = time.time()
                try:
                    if modules.get("memory_store") is not None:
                        modules["memory_store"].set_session(f"{base_session}:{name}:{qid}")
                    report = asyncio.run(run_research(query, cfg, modules))
                    elapsed = time.time() - start

                    score, score_detail = cls._score_report(report, q, evaluator)
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "elapsed_seconds": elapsed,
                        "report_length": len(report),
                        "system": name,
                        "composite_score": score,
                        "score_detail": score_detail,
                        "harness": collect_harness_telemetry(modules),
                    })
                    scores.append(score)
                    logger.info(f"    → score={score:.3f}, time={elapsed:.1f}s, len={len(report)}")

                except Exception as e:
                    logger.warning(f"    → 失败: {e}")
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "error": str(e),
                        "system": name,
                    })
                    scores.append(0.0)

            results.append({
                "system_name": name,
                "description": desc,
                "num_questions": len(questions),
                "average_composite_score": sum(scores) / len(scores) if scores else 0.0,
                "details": details,
            })

        return {
            "evaluation_name": "DeepResearch Agent 模块消融实验",
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "systems": results,
            "summary": {r["system_name"]: r["average_composite_score"] for r in results},
        }

    # -----------------------------------------------------------------------
    # 对抗轮数消融：0/1/2/3 轮
    # -----------------------------------------------------------------------
    @classmethod
    def run_rounds_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        max_rounds: int = 3,
        evaluator: Callable[[str, dict[str, Any]], float | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        在不同对抗轮数下运行评测。

        Args:
            config: 基础配置。
            questions: 评测题目列表。
            max_rounds: 最大对抗轮数。

        Returns:
            键为 adv_0 / adv_1 / ... / adv_N 的结果字典。
        """
        summary: dict[str, float] = {}
        full_details: dict[str, Any] = {}

        for rounds in range(max_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"正在运行对抗轮数 = {rounds}")
            logger.info(f"{'='*50}")

            overrides = {
                "adversarial": {
                    "max_rounds": rounds,
                    "enabled": rounds > 0,
                }
            }
            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)
            base_session = getattr(modules.get("memory_store"), "session_id", "")

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for idx, q in enumerate(questions, 1):
                qid = q.get("id", f"q{idx}")
                query = q.get("query", "")
                logger.info(f"  [{idx}/{len(questions)}] {qid}")

                try:
                    if modules.get("memory_store") is not None:
                        modules["memory_store"].set_session(f"{base_session}:adv_{rounds}:{qid}")
                    report = asyncio.run(run_research(query, cfg, modules))
                    score, score_detail = cls._score_report(report, q, evaluator)
                    scores.append(score)
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "report_length": len(report),
                        "composite_score": score,
                        "score_detail": score_detail,
                        "harness": collect_harness_telemetry(modules),
                    })
                except Exception as e:
                    logger.warning(f"    → 失败: {e}")
                    scores.append(0.0)
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "error": str(e),
                    })

            avg_score = sum(scores) / len(scores) if scores else 0.0
            key = f"adv_{rounds}"
            summary[key] = avg_score
            full_details[key] = details
            logger.info(f"对抗轮数 {rounds} 平均得分: {avg_score:.4f}")

        return {
            "evaluation_name": "DeepResearch Agent 对抗轮数消融实验",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "details": full_details,
            "config": config,
        }

    # -----------------------------------------------------------------------
    # 结果保存
    # -----------------------------------------------------------------------
    @staticmethod
    def save_results(data: dict[str, Any], output_dir: str, prefix: str = "ablation") -> str:
        """保存消融结果到 JSON 文件。"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"消融结果已保存: {filepath}")
        return filepath
