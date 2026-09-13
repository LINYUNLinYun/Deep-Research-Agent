"""
M5 Red-Blue 对抗降噪循环 — 主控制器

AdversarialLoop 驱动 Red Agent → Blue Agent → 评分的完整对抗流程，
具备死循环检测、震荡检测、收敛判断等鲁棒机制。

设计决策：
1. 维护 resolved_issues 集合：已修复的 issue 如果在新轮次中重新出现，判定为震荡。
2. 收敛条件三选一：round >= 3 / overall >= 8.0 / Δscore < 0.3。
3. 每轮记录完整评分历史，便于后续分析和审计。
"""
from __future__ import annotations

import copy
import inspect
import logging
import re
from typing import Any

from src.adversarial.blue_agent import BlueAgent
from src.adversarial.red_agent import RedAgent
from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    Issue,
    RedVerdict,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_chain


__all__ = ["AdversarialLoop"]

logger = logging.getLogger(__name__)


class AdversarialLoop:
    """Red-Blue 对抗降噪循环主控制器。

    Attributes:
        red_agent: Red Agent 实例，负责攻击。
        blue_agent: Blue Agent 实例，负责修复。
        policy: 用于 self_verify 或辅助评分的策略对象（可选）。
        max_rounds: 硬上限轮数。
        score_threshold: 综合分达标阈值。
        delta_threshold: 轮间变化量收敛阈值。
    """

    def __init__(
        self,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        policy: Any | None = None,
        max_rounds: int = 3,
        score_threshold: float = 8.0,
        delta_threshold: float = 0.3,
        rescore_after_fix: bool = True,
        evidence_verifier: Any | None = None,
    ):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.policy = policy
        self.max_rounds = max(max_rounds, 1)
        self.score_threshold = score_threshold
        self.delta_threshold = delta_threshold
        # A post-fix score is required for ``final_score`` to describe the
        # returned report rather than the pre-fix Red verdict.  The flag keeps
        # the old call shape usable for constrained/offline deployments.
        self.rescore_after_fix = bool(rescore_after_fix)
        self.evidence_verifier = evidence_verifier

    @trace_chain(name="adversarial_loop.run", tags=["m5", "loop", "adversarial"])
    async def run(
        self, report: ResearchReport
    ) -> tuple[ResearchReport, list[dict[str, Any]]]:
        """运行完整的对抗降噪循环。

        流程：
        1. 每轮用 Red Agent 攻击当前报告。
        2. Blue Agent 根据 Verdict 修复。
        3. 记录本轮评分和修复操作。
        4. 检查收敛条件或震荡/死循环。
        5. 返回最终报告和完整历史。

        Args:
            report: 初始研究报告（不会被修改，内部深拷贝）。

        Returns:
            (修复后的报告, 每轮评分记录列表)
            每轮记录包含: round, dimension_scores, overall_score, delta, issues_count,
            fix_operations, resolved_count, oscillation_detected, stop_reason
        """
        current = copy.deepcopy(report)
        best_report = copy.deepcopy(current)
        best_score: float | None = None
        accepted_rounds = 0
        history: list[dict[str, Any]] = []
        prev_scores: dict[Dimension, float] | None = None
        # Store canonical fingerprints instead of full LLM wording.  Models
        # often paraphrase the same issue on the next round; exact dataclass
        # equality made oscillation detection miss those repetitions.
        resolved_issues: set[tuple[str, str, str]] = set()
        oscillation_detected = False
        stop_reason = ""

        # Surface unavailable backends to the caller instead of converting an
        # authentication/network error into a neutral verdict or convergence.
        current.adversarial_status = "running"

        for round_idx in range(1, self.max_rounds + 1):
            logger.info(f"[AdversarialLoop] Round {round_idx} starting...")

            # ---- Step 1: Red Attack ----
            verdict = await self.red_agent.attack(current)
            if verdict.status != "success":
                reason = verdict.error or "Red Agent backend unavailable"
                logger.error(f"[Adversarial] SKIPPED: {reason}")
                stop_reason = f"red_failed_at_round_{round_idx}"
                history.append(self._build_round_record(
                    round_idx, verdict, [], len(resolved_issues), False,
                    stop_reason, accepted=False, outcome="failed",
                ))
                history[-1]["fallback_to_best"] = accepted_rounds > 0
                return self._failure_fallback(
                    best_report, best_score, accepted_rounds, round_idx, reason,
                    initial_status="skipped",
                ), history
            if best_score is None:
                best_score = verdict.overall_score
                best_report = copy.deepcopy(current)
            logger.info(
                f"[AdversarialLoop] Red attack done: overall={verdict.overall_score:.2f}, "
                f"issues={len(verdict.issues)}"
            )

            # ---- Step 2: 震荡检测 ----
            # 如果当前 issues 中有已修复过的 issue 重新出现，判定震荡
            verdict_issue_keys = {self._issue_key(issue) for issue in verdict.issues}
            reappeared = resolved_issues.intersection(verdict_issue_keys)
            if reappeared:
                oscillation_detected = True
                logger.warning(
                    f"[AdversarialLoop] Oscillation detected at round {round_idx}: "
                    f"{len(reappeared)} previously resolved issues reappeared."
                )
                stop_reason = f"oscillation_at_round_{round_idx}"
                history.append(self._build_round_record(
                    round_idx, verdict, [], len(reappeared), oscillation_detected, stop_reason
                ))
                break

            # ---- Step 3: Blue Defend ----
            fixed_report, operations = await self.blue_agent.defend(current, verdict)
            logger.info(
                f"[AdversarialLoop] Blue defend done: operations={len(operations)}"
            )
            blue_status = getattr(self.blue_agent, "status", "success")
            if blue_status == "failed" and fixed_report.content == current.content:
                reason = getattr(self.blue_agent, "error", "Blue Agent backend unavailable")
                stop_reason = f"blue_failed_at_round_{round_idx}"
                history.append(self._build_round_record(
                    round_idx, verdict, operations, len(resolved_issues), False,
                    stop_reason, accepted=False, outcome="failed",
                ))
                logger.error(f"[Adversarial] FAILED: {reason}")
                history[-1]["fallback_to_best"] = accepted_rounds > 0
                return self._failure_fallback(
                    best_report, best_score, accepted_rounds, round_idx, reason,
                    initial_status="failed",
                ), history

            # Re-score the post-fix content.  Without this pass a report could
            # be materially changed by Blue while final_score still reflected
            # Red's score for the old text.  A backend failure is surfaced as a
            # skipped adversarial run instead of being converted into a score.
            scored_verdict = verdict
            if self.rescore_after_fix and fixed_report.content != current.content:
                scored_verdict = await self.red_agent.attack(fixed_report)
                if scored_verdict.status != "success":
                    reason = scored_verdict.error or "post-fix Red Agent backend unavailable"
                    stop_reason = f"post_fix_red_failed_at_round_{round_idx}"
                    history.append(self._build_round_record(
                        round_idx, verdict, operations, len(resolved_issues), False,
                        stop_reason, accepted=False,
                        outcome="discarded",
                    ))
                    history[-1]["fallback_to_best"] = accepted_rounds > 0
                    return self._failure_fallback(
                        best_report, best_score, accepted_rounds, round_idx, reason,
                        initial_status="failed",
                    ), history

            # Never commit a round that regresses its own independent post-fix
            # Red score.  This applies to both complete and partial defenses.
            acceptance_floor = max(
                verdict.overall_score,
                best_score if best_score is not None else verdict.overall_score,
            )
            if (
                fixed_report.content != current.content
                and scored_verdict.overall_score + 1e-9 < acceptance_floor
            ):
                stop_reason = (
                    "post_fix_score_regressed"
                    f"({scored_verdict.overall_score:.3f}<{acceptance_floor:.3f})"
                )
                history.append(self._build_round_record(
                    round_idx, scored_verdict, operations, len(resolved_issues), False,
                    stop_reason, pre_verdict=verdict, accepted=False,
                    outcome="discarded",
                ))
                history[-1]["fallback_to_best"] = accepted_rounds > 0
                return self._failure_fallback(
                    best_report, best_score, accepted_rounds, round_idx, stop_reason,
                    initial_status="rejected",
                ), history

            # Only accepted rounds may influence oscillation state.
            for op in operations:
                if op.success:
                    resolved_issues.add(self._issue_key(op.issue))

            # ---- Step 4: 计算 delta ----
            delta = 0.0
            if prev_scores is not None:
                delta = VerdictEngine.compute_delta(prev_scores, scored_verdict.dimension_scores)
            prev_scores = copy.deepcopy(scored_verdict.dimension_scores)

            # ---- Step 5: 记录本轮 ----
            stop_reason = self._check_convergence(round_idx, scored_verdict.overall_score, delta)
            if blue_status == "partial":
                stop_reason = f"blue_partial_at_round_{round_idx}"
            record = self._build_round_record(
                round_idx=round_idx,
                verdict=scored_verdict,
                operations=operations,
                resolved_count=len(resolved_issues),
                oscillation=oscillation_detected,
                stop_reason=stop_reason,
                pre_verdict=verdict if scored_verdict is not verdict else None,
                outcome="partial" if blue_status == "partial" else "success",
            )
            history.append(record)

            # 更新当前报告为修复后的版本
            changed = fixed_report.content != current.content
            current = fixed_report
            current.adversarial_rounds = round_idx
            if blue_status == "partial":
                current.adversarial_status = "partial"
                current.adversarial_reason = getattr(self.blue_agent, "error", "")

            evidence_summary: dict[str, Any] | None = None
            if self.evidence_verifier is not None:
                try:
                    verify_fn = getattr(self.evidence_verifier, "verify", None)
                    if verify_fn is None:
                        verify_fn = getattr(self.evidence_verifier, "verify_sync")
                    verification = verify_fn(current)
                    if inspect.isawaitable(verification):
                        verification = await verification
                    serialized_verification = [
                        item.to_dict() if hasattr(item, "to_dict") else item
                        for item in (verification or [])
                    ]
                    summarize = getattr(self.evidence_verifier, "summary", None)
                    if callable(summarize):
                        evidence_summary = summarize(verification or [])
                    current.claim_evidence_edges = [
                        {
                            "claim_id": item.get("claim", {}).get("claim_id", ""),
                            "source_id": evidence.get("metadata", {}).get("source_id", ""),
                            "relation": item.get("status", "unknown"),
                            "confidence": item.get("confidence", 0.0),
                            "reason": item.get("reason", ""),
                        }
                        for item in serialized_verification
                        if isinstance(item, dict)
                        for evidence in item.get("evidence", [])
                        if isinstance(evidence, dict)
                    ]
                    current.evidence_verification = evidence_summary or {
                        "results": serialized_verification
                    }
                except Exception as exc:  # evidence is a quality signal, not a hard failure
                    logger.warning("[AdversarialLoop] evidence verification failed: %s", exc)
                    current.evidence_verification = {"error": str(exc)}
                    evidence_summary = {"error": str(exc)}
            if evidence_summary is not None:
                record["evidence_summary"] = evidence_summary

            if changed:
                accepted_rounds += 1
                best_score = scored_verdict.overall_score
                best_report = copy.deepcopy(current)
            record["accepted_rounds"] = accepted_rounds

            # ---- Step 6: 判断是否停止 ----
            if stop_reason:
                logger.info(f"[AdversarialLoop] Stopping: {stop_reason}")
                break

        # 循环结束后写入最终分数
        if history:
            current.final_score = history[-1]["overall_score"]
            if current.adversarial_status == "running":
                current.adversarial_status = "success"
                current.adversarial_reason = ""
        elif current.adversarial_status == "running":
            current.adversarial_status = "success"

        return current, history

    @staticmethod
    def _failure_fallback(
        best_report: ResearchReport,
        best_score: float | None,
        accepted_rounds: int,
        round_idx: int,
        reason: str,
        *,
        initial_status: str,
    ) -> ResearchReport:
        """Return the last accepted report when a later round fails.

        A failed attempt must not invalidate or ambiguously relabel content
        committed by an earlier round. With no accepted round, preserve the
        original failure status and original report.
        """
        result = copy.deepcopy(best_report)
        result.adversarial_status = "partial" if accepted_rounds else initial_status
        result.adversarial_reason = reason
        result.adversarial_rounds = round_idx
        if best_score is not None:
            result.final_score = best_score
        return result

    @staticmethod
    def _issue_key(issue: Issue) -> tuple[str, str, str]:
        """Canonical issue fingerprint used for oscillation detection."""
        dimension = getattr(issue.dimension, "value", str(issue.dimension))
        fix_type = getattr(issue.fix_type, "value", str(issue.fix_type))
        location = re.sub(r"\s+", " ", (issue.location or "").strip().lower())
        if not location:
            # Keep a compact description anchor when the model omitted a
            # location; remove volatile numbers so paraphrased reports match.
            description = re.sub(r"\d+(?:\.\d+)?", "#", issue.description or "")
            location = " ".join(description.lower().split())[:120]
        return dimension, location, fix_type

    def _check_convergence(
        self, round_idx: int, overall_score: float, delta: float
    ) -> str:
        """检查是否满足任一收敛条件。

        Returns:
            空字符串表示继续；非空字符串为停止原因。
        """
        if round_idx >= self.max_rounds:
            return f"max_rounds_reached({self.max_rounds})"
        if overall_score >= self.score_threshold:
            return f"score_threshold_met({overall_score:.2f}>={self.score_threshold})"
        # 第一轮没有上一轮做对比，delta 恒为 0，跳过 delta 收敛检测
        if round_idx > 1 and delta < self.delta_threshold:
            return f"delta_converged({delta:.3f}<{self.delta_threshold})"
        return ""

    def _build_round_record(
        self,
        round_idx: int,
        verdict: RedVerdict,
        operations: list[FixOperation],
        resolved_count: int,
        oscillation: bool,
        stop_reason: str,
        pre_verdict: RedVerdict | None = None,
        accepted: bool = True,
        outcome: str = "success",
    ) -> dict[str, Any]:
        """构造单轮记录字典。"""
        record = {
            "round": round_idx,
            "dimension_scores": {
                k.value: round(v, 3) for k, v in verdict.dimension_scores.items()
            },
            "overall_score": round(verdict.overall_score, 3),
            "issues_count": len(verdict.issues),
            "issue_stats": dict(getattr(verdict, "issue_stats", {}) or {}),
            "red_retry_stats": dict(getattr(verdict, "retry_stats", {}) or {}),
            "fix_operations": [op.to_dict() for op in operations],
            "resolved_count": resolved_count,
            "oscillation_detected": oscillation,
            "stop_reason": stop_reason,
            "accepted": accepted,
            "outcome": outcome,
            "raw_feedback": verdict.raw_feedback,
        }
        if operations:
            record["blue_repair_stats"] = dict(
                getattr(self.blue_agent, "last_repair_stats", {}) or {}
            )
        if pre_verdict is not None:
            record["pre_fix_score"] = round(pre_verdict.overall_score, 3)
            record["post_fix_score"] = round(verdict.overall_score, 3)
            record["pre_fix_dimension_scores"] = {
                k.value: round(v, 3) for k, v in pre_verdict.dimension_scores.items()
            }
        return record
