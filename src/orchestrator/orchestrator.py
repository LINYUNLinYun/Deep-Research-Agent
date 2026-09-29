"""
Deep Research Agent — 核心编排器 (M1: Multi-Agent Orchestrator)

9 状态状态机驱动的异步任务编排引擎：
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
  失败时进入 REPLANNING，最终可进入 FAILED。

设计亮点:
  - 自研 asyncio + DAG executor，不依赖 LangGraph/AutoGen
  - 拓扑排序后按层并发执行，Semaphore 控制最大并发度
  - 三级降级策略：单任务超时→标记继续；>50%失败→re-plan；全局超时→强制合成
  - 状态机用字典映射实现，便于扩展新状态和转换逻辑
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import replace
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
    ResearchState,
    DecisionRecord,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..planner.planner import Planner, PlanParseError
from ..planner.budget_tracker import BudgetTracker
from ..utils.tracing import trace_chain

# M4: Memory Store 类型提示（延迟导入避免循环依赖）
SharedMemoryStore = Any


__all__ = ["Orchestrator"]


class Orchestrator:
    """Deep Research Agent 核心编排器。

    Attributes:
        planner: 自适应规划器，负责初始规划和增量重规划。
        agent_pool: Agent 对象池，管理 worker agent 生命周期。
        budget_tracker: Token 预算追踪器。
        memory_store: 全局共享内存，存储所有子任务结果和中间上下文。
        compressor: （预留）上下文压缩器接口。
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        budget_tracker: BudgetTracker | None = None,
        compressor: Any | None = None,
        adversarial_loop: Any | None = None,
        memory_store: Any | None = None,
        summarizer_policy: Any | None = None,
        evidence_verifier: Any | None = None,
        max_catalog_sources: int = 24,
        max_catalog_chars: int = 12_000,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy
        self.evidence_verifier = evidence_verifier
        self.max_catalog_sources = max_catalog_sources
        self.max_catalog_chars = max_catalog_chars

        # 运行时状态（保留 dict 作为快速缓存，M4 提供持久化 + 语义检索）
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._historical_results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        self._replan_count: int = 0
        self._adversarial_count: int = 0
        self._decision_trace: list[DecisionRecord] = []
        self._seen_evidence: set[str] = set()
        self._low_novelty_rounds: int = 0
        self._pending_replan_reason: str = ""
        self._reusable_results: dict[str, AgentResult] = {}
        self._research_graph: Any | None = None
        self._facet_task_outcomes: dict[str, dict[str, AgentResult]] = {}

        # 状态机处理器映射
        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.SYNTHESIZING: self._do_synthesizing,
            OrchestratorState.ADVERSARIAL: self._do_adversarial,
            OrchestratorState.REPLANNING: self._do_replanning,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="orchestrator.run", tags=["m1", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        """主入口：执行完整的研究流程。

        Args:
            query: 研究问题。
            config: 运行配置，默认使用 RunConfig()。

        Returns:
            ResearchReport: 最终研究报告。
        """
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._decision_trace.clear()
        self._seen_evidence.clear()
        self._low_novelty_rounds = 0
        self._pending_replan_reason = ""
        self._reusable_results.clear()
        self._research_graph = None
        self._facet_task_outcomes.clear()
        self._memory_store.clear()
        self._results.clear()
        self._historical_results.clear()
        self._dag = None
        self._task_map.clear()
        self._current_state = OrchestratorState.IDLE

        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            # 全局超时检查
            if self._is_global_timeout():
                if self._results or self._historical_results:
                    self._force_synthesize_from_partial("global_timeout")
                    self._current_state = OrchestratorState.DONE
                else:
                    self._current_state = OrchestratorState.FAILED
                break

            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            next_state = await handler()
            self._current_state = next_state

            print(f"[Orchestrator] State transition: {self._current_state.value}")

        # 返回结果
        if self._current_state == OrchestratorState.DONE:
            # 最终报告应在 memory 中
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count
            report.decision_trace = [record.to_dict() for record in self._decision_trace]
            if self._research_graph is not None:
                report.research_state = self._research_graph.to_dict()

            # M4: 将最终报告存入 SharedMemoryStore
            if self.memory_store is not None:
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"final_report:{int(time.time())}",
                        claim=str(report.content)[:800],
                        source="orchestrator",
                        confidence=report.confidence,
                        agent_id="orchestrator",
                        timestamp=time.time(),
                        evidence_type="primary",
                        embedding=[],
                        topic=query[:50],
                        metadata={
                            "num_searches": report.num_searches,
                            "num_replan": report.num_replan,
                            "adversarial_rounds": report.adversarial_rounds,
                        },
                    )
                    self.memory_store.put(entry)
                    print(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")
                except Exception as e:
                    print(f"[M4] Failed to store final report: {e}")

            return report

        # FAILED 状态
        return ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
        )

    # ------------------------------------------------------------------
    # 状态机处理器
    # ------------------------------------------------------------------

    async def _on_idle(self) -> OrchestratorState:
        """从 IDLE 自动进入 PLANNING。"""
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """调用 Planner 生成初始 DAG。

        失败时直接转入 FAILED（初始计划失败无法恢复）。
        """
        try:
            memory_ctx = self._build_memory_context()
            self._dag = self.planner.generate_plan(self._query, memory_ctx)
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
            self._initialize_research_state()
        except PlanParseError as e:
            print(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        print(f"[Planning] ✓ DAG 生成完成: {n_tasks} 个子任务, {n_layers} 个执行层")
        # 打印子任务描述以便诊断
        for tid, task in self._task_map.items():
            print(f"[Planning]   {tid}: {task.description}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """拓扑排序 + 并发调度 sub-agents。

        核心逻辑:
          1. 获取并行执行层 (parallel groups)
          2. 每层内用 asyncio.gather + Semaphore 并发执行
          3. 每个 sub-task 设置单独超时 (asyncio.wait_for)
          4. 收集结果到 self._results
        """
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")

            # 构建本层的 coroutine 列表
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    signature = self._task_signature(subtask)
                    reusable = self._reusable_results.get(signature)
                    if reusable is not None and reusable.status == AgentStatus.SUCCESS:
                        return replace(
                            reusable,
                            task_id=task_id,
                            trajectory=list(reusable.trajectory) + [{
                                "event": "reused",
                                "signature": signature,
                                "original_task_id": reusable.task_id,
                            }],
                        )

                    missing = []
                    for dep_id in subtask.dependencies:
                        dep = self._memory_store.get(f"result:{dep_id}")
                        if dep is None or dep.status != AgentStatus.SUCCESS:
                            missing.append(dep_id)
                    if missing:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Blocked by failed or missing dependencies: {', '.join(missing)}",
                            trajectory=[{"event": "dependency_blocked", "dependencies": missing}],
                        )

                    # 准备上下文：先执行依赖任务的结果
                    context = self._build_task_context(subtask)

                    # 获取 Agent
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    route_trace = getattr(agent.policy, "decision_trace", [])
                    route_start = len(route_trace) if isinstance(route_trace, list) else 0
                    result: AgentResult | None = None
                    try:
                        # 设置单任务超时
                        result = await asyncio.wait_for(
                            agent.run(subtask, context),
                            timeout=subtask.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {subtask.timeout_seconds}s",
                        )
                    except Exception as e:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        route_trace = getattr(agent.policy, "decision_trace", [])
                        if result is not None and isinstance(route_trace, list):
                            for route in route_trace[route_start:]:
                                if isinstance(route, dict):
                                    result.trajectory.append({"event": "model_route", **route})
                                    self._decision_trace.append(DecisionRecord(
                                        action="model_route",
                                        signals={"task_id": task_id, **route},
                                        model=str(route.get("backend", "")),
                                        latency=float(route.get("latency", 0.0) or 0.0),
                                        reason=str(route.get("status", "")),
                                        timestamp=time.time(),
                                    ))
                        if result is not None:
                            for event in result.trajectory:
                                if not isinstance(event, dict) or event.get("role") != "tool":
                                    continue
                                payload = event.get("result")
                                policy_record = payload.get("_search_policy") if isinstance(payload, dict) else None
                                if not isinstance(policy_record, dict):
                                    continue
                                artifact = dict(policy_record.get("artifact", {}) or {})
                                self._decision_trace.append(DecisionRecord(
                                    action=str(policy_record.get("action", "search_policy")),
                                    signals={
                                        "task_id": task_id,
                                        "decision_point": policy_record.get("decision_point", "after_search"),
                                        "rule_id": policy_record.get("rule_id", ""),
                                        "policy_signals": policy_record.get("signals", {}),
                                        "artifact": artifact,
                                    },
                                    reason=str(policy_record.get("reason", "")),
                                    timestamp=time.time(),
                                ))
                        await self.agent_pool.release_agent(agent)

                    return result or AgentResult(
                        task_id=task_id,
                        status=AgentStatus.FAILED,
                        output="Agent execution was cancelled before producing a result",
                    )

            # 并发执行本层
            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # 将异常包装为 FAILED 结果
                    # 这种情况理论上不会发生（_run_one 内部已捕获），但保险起见
                    all_results.append(AgentResult(
                        task_id="unknown",
                        status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}",
                    ))
                else:
                    all_results.append(lr)

            # Commit at the layer barrier. This is the crucial DAG data-flow
            # invariant: the next layer can now consume its predecessors.
            for result in all_results[-len(layer_results):]:
                self._memory_store[f"result:{result.task_id}"] = result
                subtask = self._task_map.get(result.task_id)
                if subtask is not None and result.status == AgentStatus.SUCCESS:
                    self._reusable_results[self._task_signature(subtask)] = result
            self._update_research_state_layer(all_results[-len(layer_results):], layer_idx)

        self._results = all_results
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集结果，写入 memory，检查是否需要重规划。

        三级降级策略检查点:
          - 单任务超时/失败：已在 dispatch 层处理（标记状态，继续执行）
          - >50% 失败：触发 REPLANNING
          - 全局超时：由外层 run() 的循环检查处理
        """
        # 将结果写入运行时 memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r
            self.budget_tracker.track(max(0, int(getattr(r, "token_usage", 0))))

        # M4: 将成功结果同步写入 SharedMemoryStore（持久化 + 向量索引）
        if self.memory_store is not None:
            for r in self._results:
                if r.status == AgentStatus.SUCCESS and r.output:
                    self._sync_result_to_memory_store(r)

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        failed_ids = [r.task_id for r in self._results if r.status != AgentStatus.SUCCESS]
        status_icon = "✓" if success_count == total_count else "⚠"
        print(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功", end="")
        if fail_count > 0:
            print(f" ({fail_count} 失败)")
        else:
            print()

        decision = self._decide_after_collection(self._results)
        print(
            f"[Replan] trigger check: failed_count={fail_count}, "
            f"replan_enabled={self._config.max_replan_rounds > 0}, "
            f"current_replans={self._replan_count}, max_replans={self._config.max_replan_rounds}, "
            f"decision={decision.action}, failed={failed_ids}"
        )

        # 检查是否需要重规划
        if decision.action == "replan":
            if self._replan_count < self._config.max_replan_rounds:
                self._replan_count += 1
                return OrchestratorState.REPLANNING
            else:
                print("[Collect] Max replan rounds reached, proceeding with partial results")
                # 超过最大重规划次数，继续合成（用已有结果）

        return OrchestratorState.SYNTHESIZING

    def _sync_result_to_memory_store(self, result: AgentResult) -> None:
        """将 AgentResult 同步到 M4 SharedMemoryStore。

        提取 output 中的关键 claim 作为记忆条目，支持后续语义检索。
        """
        try:
            # 延迟导入避免循环依赖
            from src.memory.long_term import MemoryEntry
            claim_text = str(result.output)[:500]  # 取前 500 字作为 claim
            session = getattr(self.memory_store, "session_id", "")
            entry = MemoryEntry(
                entry_id=f"{session}:{result.task_id}:{int(time.time() * 1000)}",
                claim=claim_text,
                source=f"task:{result.task_id}",
                confidence=getattr(result, "confidence", 0.5),
                agent_id=result.task_id,
                timestamp=time.time(),
                evidence_type="primary",
                embedding=[],  # SharedMemoryStore.put() 会自动生成 embedding
                topic=self._query[:50],
                metadata={
                    "status": result.status.value,
                    "token_usage": getattr(result, "token_usage", 0),
                    "sources": self._extract_sources(result),
                    "verification_status": "unverified",
                },
            )
            self.memory_store.put(entry)
            print(f"[M4] Memory stored: {result.task_id} (claim={claim_text[:60]}...)")
        except Exception as e:
            print(f"[M4] Failed to store memory for {result.task_id}: {e}")

    async def _do_synthesizing(self) -> OrchestratorState:
        """调用 SummarizerAgent 合成研究报告。"""
        # 创建合成任务
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,  # 使用 ANALYZE 类型，实际由 SummarizerAgent 处理
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        context = {
            "query": self._query,
            "results": self._historical_results + self._results,
            "prior_report": self._memory_store.get("prior_report"),
        }
        # Synthesis has a dedicated policy and should not borrow/leak a worker
        # from the analyze pool.
        from ..agents.summarizer import SummarizerAgent
        policy = self.summarizer_policy
        if policy is None:
            borrowed = await self.agent_pool.get_agent(TaskType.ANALYZE)
            policy = borrowed.policy
            tools = borrowed.tools
            await self.agent_pool.release_agent(borrowed)
        else:
            tools = []
        agent = SummarizerAgent(
            name="summarizer",
            policy=policy,
            tools=tools,
            max_catalog_sources=self.max_catalog_sources,
            max_catalog_chars=self.max_catalog_chars,
        )

        # Freeze provenance before any L3 aggregation, but pass only a ranked,
        # token-bounded projection to the synthesizer. The lossless bundles on
        # AgentResult remain available for later verification.
        context["source_catalog"] = agent.collect_sources(self._query, context["results"])

        if self.compressor is not None:
            context["results"] = self._compress_results_for_context(context["results"])

        try:
            result = await asyncio.wait_for(
                agent.run(synth_task, context),
                timeout=synth_task.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.TIMEOUT,
                output="Synthesis timed out",
            )
        except Exception as e:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.FAILED,
                output=f"Synthesis error: {type(e).__name__}: {e}",
            )

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            self._memory_store["final_report"] = result.output
        else:
            # 合成失败但已有结果，生成降级报告
            self._memory_store["final_report"] = ResearchReport(
                query=self._query,
                content=str(result.output) if result.output else "Synthesis failed.",
                confidence=0.0,
                num_searches=sum(
                    len([t for t in r.trajectory if t.get("role") == "tool"])
                    for r in (self._historical_results + self._results)
                ),
            )

        verification_summary = await self._verify_report_evidence(
            self._memory_store["final_report"]
        )
        if (
            verification_summary
            and verification_summary.get("unsupported_rate", 0.0)
            >= self._config.evidence_replan_threshold
            and self._config.enable_replan
            and self._replan_count < self._config.max_replan_rounds
        ):
            targeted_tasks = self._prepare_evidence_gap_tasks(verification_summary)
            if targeted_tasks:
                self._replan_count += 1
            self._decision_trace.append(DecisionRecord(
                action="targeted_verify" if targeted_tasks else "synthesize",
                signals={
                    **verification_summary,
                    "targeted_task_ids": targeted_tasks,
                },
                reason=(
                    "unsupported claims scheduled as a bounded verification DAG"
                    if targeted_tasks
                    else "no actionable unresolved claims; skip broad replan"
                ),
                timestamp=time.time(),
            ))
            if targeted_tasks:
                self._memory_store["prior_report"] = self._memory_store["final_report"]
                return OrchestratorState.DISPATCHING

        if self._config.enable_adversarial:
            print("[Synthesize] ✓ 报告合成完成，进入对抗优化")
            return OrchestratorState.ADVERSARIAL
        print("[Synthesize] ✓ 报告合成完成")
        return OrchestratorState.DONE

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue 对抗降噪循环。

        调用 AdversarialLoop 对报告进行 challenge-verify 迭代优化。
        仅在报告置信度低于阈值时触发，避免资源浪费。
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE

        # 置信度足够高时跳过对抗
        if report.confidence >= self._config.adversarial_confidence_threshold:
            print("[Adversarial] ✓ 报告置信度达到动态配置阈值，跳过对抗优化")
            report.adversarial_status = "skipped"
            report.adversarial_reason = "report_confidence_above_threshold"
            return OrchestratorState.DONE

        if self.adversarial_loop is None:
            print("[Adversarial] AdversarialLoop 未配置，跳过")
            report.adversarial_status = "skipped"
            report.adversarial_reason = "adversarial_backend_not_configured"
            return OrchestratorState.DONE

        try:
            print(f"[Adversarial] ▶ 启动 Red-Blue 对抗优化 (当前置信度={report.confidence:.2f})")
            elapsed = time.monotonic() - self._start_time
            remaining_global = self._config.global_timeout_seconds - elapsed
            timeout = min(self._config.adversarial_timeout_seconds, remaining_global)
            if timeout <= 0:
                raise asyncio.TimeoutError("global timeout reached before adversarial stage")
            optimized_report, history = await asyncio.wait_for(
                self.adversarial_loop.run(report), timeout=timeout
            )
            self._memory_store["final_report"] = optimized_report
            self._adversarial_count += len(history)
            if getattr(optimized_report, "adversarial_status", "success") in {"skipped", "failed", "rejected"}:
                print(f"[Adversarial] SKIPPED: {optimized_report.adversarial_reason}")
            else:
                print(f"[Adversarial] ✓ 对抗优化完成: {len(history)} 轮, 最终置信度={optimized_report.confidence:.2f}")
        except asyncio.TimeoutError:
            report.adversarial_status = "skipped"
            report.adversarial_reason = "adversarial_timeout"
            report.adversarial_rounds = 0
            print("[Adversarial] SKIPPED: adversarial_timeout，使用原始报告")
        except Exception as e:
            report.adversarial_status = "skipped"
            report.adversarial_reason = str(e)
            report.adversarial_rounds = 0
            print(f"[Adversarial] SKIPPED: {e}，使用原始报告")

        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """触发增量重规划。

        保留 confidence≥0.6 的成功结果，修改失败子问题。
        """
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        reason = self._pending_replan_reason or self._build_failure_reason(self._results)
        self._pending_replan_reason = ""
        print(f"[Replan] Round {self._replan_count}/{self._config.max_replan_rounds}. Failed tasks: {[t.task_id for t in failed_tasks]}")

        all_existing_results = self._historical_results + self._results
        usable_confidences = [
            r.confidence for r in all_existing_results
            if r.status == AgentStatus.SUCCESS
        ]
        preserve_threshold = self._adaptive_preserve_threshold(usable_confidences)

        try:
            new_dag = self.planner.replan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=all_existing_results,
                reason=reason,
                preserve_threshold=preserve_threshold,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            self._extend_research_state()
            # Preserve successful evidence for final synthesis while the next
            # dispatch round gets a clean result set for failure accounting.
            self._historical_results.extend(
                r for r in self._results
                if r.status == AgentStatus.SUCCESS and r.task_id not in {x.task_id for x in self._historical_results}
            )
            # 清空上一轮结果（保留在 memory 中，新任务可通过 context_keys 引用）
            self._results = []
        except PlanParseError as e:
            print(f"[Replan] Failed: {e}")
            # 重规划失败，如果已有部分成功结果，尝试直接合成
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Replan] Unexpected error: {e}")
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED

        return OrchestratorState.DISPATCHING

    async def _on_done(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # 决策逻辑
    # ------------------------------------------------------------------

    def _should_replan(self, results: list[AgentResult]) -> bool:
        """判断是否需要重规划。

        策略：任何明确失败/超时的子任务都触发有限重规划；上限由
        RunConfig 控制，避免局部失败静默进入最终报告。
        """
        return self._decide_after_collection(results, record=False).action == "replan"

    def _decide_after_collection(
        self, results: list[AgentResult], record: bool = True
    ) -> DecisionRecord:
        total = len(results)
        failed_results = [r for r in results if r.status != AgentStatus.SUCCESS]
        successes = [r for r in results if r.status == AgentStatus.SUCCESS]
        failure_ratio = len(failed_results) / max(total, 1)
        new_keys = self._evidence_keys(successes) - self._seen_evidence
        novelty = len(new_keys) / max(len(successes), 1)
        if record:
            self._seen_evidence.update(new_keys)
            if successes and novelty < self._config.replan_min_novelty:
                self._low_novelty_rounds += 1
            else:
                self._low_novelty_rounds = 0

        critical_failures = [
            r.task_id for r in failed_results
            if self._dag is not None and bool(self._dag.get_successors(r.task_id))
        ]
        usable = [r for r in successes if r.confidence >= self._config.min_usable_confidence]
        remaining = max(0, self._config.token_budget - self.budget_tracker.get_usage())
        state = ResearchState(
            coverage=len(successes) / max(total, 1),
            evidence_novelty=novelty,
            failures=[{"task_id": r.task_id, "status": r.status.value, "error": str(r.output)[:240]} for r in failed_results],
            remaining_budget=remaining,
            successful_tasks=len(successes),
            total_tasks=total,
        )

        should_replan = (
            self._config.enable_replan
            and bool(failed_results)
            and remaining > 0
            and (
                failure_ratio >= self._config.replan_failure_ratio
                or bool(critical_failures)
                or not usable
            )
            and self._low_novelty_rounds < self._config.replan_novelty_patience
        )
        action = "replan" if should_replan else "synthesize"
        reason = (
            "failed tasks affect coverage or downstream dependencies"
            if should_replan
            else "evidence is usable, marginal gain is low, or replan budget is disabled"
        )
        decision = DecisionRecord(
            action=action,
            signals={
                "coverage": state.coverage,
                "failure_ratio": failure_ratio,
                "critical_failures": critical_failures,
                "evidence_novelty": novelty,
                "low_novelty_rounds": self._low_novelty_rounds,
                "remaining_budget": remaining,
            },
            reason=reason,
            timestamp=time.time(),
        )
        if record:
            self._decision_trace.append(decision)
        return decision

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """检查是否超过全局超时。"""
        elapsed = time.monotonic() - self._start_time
        return elapsed > self._config.global_timeout_seconds

    def _build_memory_context(self) -> str:
        """构建给 planner 的上下文摘要。

        优先使用 M4 SharedMemoryStore 的语义检索（如果已接入），
        否则回退到运行时 dict 遍历。
        """
        # M4: 语义检索相关记忆
        if self.memory_store is not None:
            try:
                ctx = self.memory_store.get_context_for_query(
                    self._query, max_tokens=2000
                )
                if ctx:
                    print(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    return ctx
            except Exception as e:
                print(f"[M4] Semantic memory query failed: {e}, falling back to dict")

        # 回退：运行时 dict 遍历
        parts = []
        for key, value in self._memory_store.items():
            if key.startswith("result:"):
                continue
            parts.append(f"{key}: {str(value)[:200]}")

        # M3: 如果上下文过长，启用压缩
        if self.compressor is not None and parts:
            total_chars = sum(len(p) for p in parts)
            if total_chars > 6000:  # 约 2000 tokens 的启发式阈值
                try:
                    compressed = self.compressor.compress(
                        texts=parts,
                        query=self._query,
                        system_prompt_tokens=0,
                    )
                    print(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    print(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """为单个 SubTask 构建执行上下文。"""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
        facet_id = subtask.facet_id or subtask.task_id
        if self._research_graph is not None:
            facet = self._research_graph.facets.get(facet_id)
            if facet is not None:
                ctx["facet"] = facet.description
                ctx["source_cluster_ids"] = sorted({
                    cluster
                    for result in self._facet_task_outcomes.get(facet_id, {}).values()
                    for cluster in self._result_source_clusters(result)
                })
                score = self._research_graph.score_details("search_new_facet", facet_id)
                ctx["frontier_action"] = "search_new_facet"
                ctx["frontier_estimated_value"] = score.value if score is not None else 0.0
        # 注入依赖任务的结果
        for dep_id in subtask.dependencies:
            dep_key = f"result:{dep_id}"
            if dep_key in self._memory_store:
                ctx[f"dep:{dep_id}"] = self._memory_store[dep_key]
        if self.compressor is not None:
            compressible = [
                f"{key}: {value.output if isinstance(value, AgentResult) else value}"
                for key, value in ctx.items()
                if key.startswith("dep:") or key in subtask.context_keys
            ]
            if compressible:
                try:
                    ctx["compressed_context"] = "\n".join(
                        self.compressor.compress(compressible, query=subtask.description)
                    )
                except Exception as exc:
                    print(f"[M3] Worker context compression failed: {exc}")
        return ctx

    def _initialize_research_state(self) -> None:
        """Create Strategy-2's deterministic coverage graph in shadow mode."""
        if not self._config.research_state_enabled:
            return
        from ..planner.research_state import ResearchStateGraph

        self._research_graph = ResearchStateGraph.from_subtasks(
            self._query,
            list(self._task_map.values()),
            dag=self._dag,
            budget_limit=max(
                1,
                self._config.max_sub_questions * (1 + self._config.max_replan_rounds),
            ),
            max_consecutive_actions=self._config.frontier_max_consecutive_action,
            marginal_gain_threshold=self._config.frontier_marginal_gain_threshold,
        )
        self._decision_trace.append(DecisionRecord(
            action="research_state_initialized",
            signals={
                "mode": "active" if self._config.research_state_active else "shadow",
                "facets": sorted(self._research_graph.facets),
                "budget_limit": self._research_graph.budget_limit,
            },
            reason="strategy2 facet graph initialized from executable DAG",
            timestamp=time.time(),
        ))

    def _update_research_state_layer(
        self,
        results: list[AgentResult],
        layer_idx: int,
    ) -> None:
        """Update facet coverage at a DAG barrier and trace the best frontier.

        The first implementation is deliberately shadow-only: it observes the
        exact production execution and never adds calls or changes DAG order.
        """
        graph = self._research_graph
        if graph is None:
            return
        for result in results:
            task = self._task_map.get(result.task_id)
            if task is None:
                continue
            facet_id = task.facet_id or task.task_id
            self._facet_task_outcomes.setdefault(facet_id, {})[task.task_id] = result

        for facet_id, outcomes in sorted(self._facet_task_outcomes.items()):
            if facet_id not in graph.facets:
                continue
            expected = [
                task for task in self._task_map.values()
                if (task.facet_id or task.task_id) == facet_id
            ]
            successes = [r for r in outcomes.values() if r.status == AgentStatus.SUCCESS]
            coverage = len(successes) / max(len(expected), 1)
            support = (
                sum(max(0.0, min(1.0, r.confidence)) for r in successes) / len(successes)
                if successes else 0.0
            )
            clusters = {
                cluster for result in successes
                for cluster in self._result_source_clusters(result)
            }
            graph.update_facet(
                facet_id,
                coverage=coverage,
                support=support,
                source_diversity=min(1.0, len(clusters) / 2.0),
            )

        selected = graph.choose_action(commit=False)
        gate = graph.stop_gate()
        self._decision_trace.append(DecisionRecord(
            action="frontier_shadow",
            signals={
                "layer": layer_idx + 1,
                "selected_action": selected.action.value,
                "target_id": selected.target_id,
                "estimated_value": round(selected.value, 6),
                "candidates": graph.frontier_scores(),
                "stop_gate": gate,
            },
            reason=selected.reason,
            timestamp=time.time(),
        ))

    def _extend_research_state(self) -> None:
        """Merge replan facets into the existing episode without losing state."""
        graph = self._research_graph
        if graph is None:
            return
        from ..planner.research_state import ResearchStateGraph

        addition = ResearchStateGraph.from_subtasks(
            self._query,
            list(self._task_map.values()),
            dag=self._dag,
            budget_limit=graph.budget_limit,
        )
        for facet_id, facet in addition.facets.items():
            if facet_id not in graph.facets:
                graph.add_facet(facet)
                continue
            current = graph.facets[facet_id]
            current.expected_questions = sorted(set(
                current.expected_questions + facet.expected_questions
            ))
            current.dependencies = sorted(set(current.dependencies + facet.dependencies))
        for claim_id, claim in addition.claims.items():
            if claim_id not in graph.claims:
                graph.add_claim(claim)
        for question_id, question in addition.open_questions.items():
            if question_id not in graph.open_questions:
                graph.add_open_question(question)

    @staticmethod
    def _result_source_clusters(result: AgentResult) -> set[str]:
        bundle = result.evidence_bundle if isinstance(result.evidence_bundle, dict) else {}
        sources = bundle.get("sources", []) if isinstance(bundle, dict) else []
        clusters = {
            str(source.get("source_cluster_id") or source.get("domain_cluster") or source.get("url") or "")
            for source in sources
            if isinstance(source, dict) and (
                source.get("source_cluster_id") or source.get("domain_cluster") or source.get("url")
            )
        }
        if clusters:
            return clusters
        return {
            str(source.get("url", ""))
            for source in Orchestrator._extract_sources(result)
            if source.get("url")
        }

    def _compress_results_for_context(self, results: list[AgentResult]) -> list[AgentResult]:
        """Compress synthesis inputs without mutating canonical AgentResults."""
        texts = [str(r.output) for r in results]
        if not texts:
            return results
        try:
            compressed = self.compressor.compress(texts, query=self._query)
        except Exception as exc:
            print(f"[M3] Synthesis compression failed: {exc}")
            return results
        # L3 may aggregate many inputs into one. In that case preserve it as a
        # synthetic evidence result rather than silently misaligning task IDs.
        if len(compressed) != len(results):
            return [AgentResult(
                task_id="compressed_evidence",
                status=AgentStatus.SUCCESS,
                output="\n\n".join(compressed),
                confidence=min((r.confidence for r in results if r.status == AgentStatus.SUCCESS), default=0.5),
            )]
        return [replace(result, output=text) for result, text in zip(results, compressed)]

    @staticmethod
    def _task_signature(task: SubTask) -> str:
        """Identity used to reuse an unchanged successful task after replan."""
        material = "\n".join([
            task.task_type.value,
            " ".join(task.description.lower().split()),
            task.expected_type,
            "|".join(sorted(task.search_hints or [])),
            "|".join(sorted(task.dependencies or [])),
            "|".join(sorted(task.context_keys or [])),
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_sources(result: AgentResult) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for step in result.trajectory:
            payload = step.get("result") if isinstance(step, dict) else None
            if not isinstance(payload, dict):
                continue
            for item in payload.get("results", []) or payload.get("papers", []):
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("pdf_url")
                if url:
                    sources.append({
                        "url": url,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet") or item.get("summary", ""),
                    })
        return sources

    def _evidence_keys(self, results: list[AgentResult]) -> set[str]:
        keys: set[str] = set()
        for result in results:
            sources = self._extract_sources(result)
            if sources:
                keys.update(source["url"] for source in sources)
            elif result.output:
                normalized = " ".join(str(result.output).lower().split())
                keys.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        return keys

    def _adaptive_preserve_threshold(self, confidences: list[float]) -> float:
        if not confidences:
            return self._config.min_usable_confidence
        ordered = sorted(confidences)
        median = ordered[len(ordered) // 2]
        return round(max(0.4, min(0.75, median - 0.1)), 2)

    def _force_synthesize_from_partial(self, reason: str) -> None:
        """Deterministic zero-call fallback used once the hard deadline expires."""
        results = self._historical_results + self._results
        usable = [r for r in results if r.status == AgentStatus.SUCCESS and r.output]
        sections = [f"## Partial result: {r.task_id}\n\n{r.output}" for r in usable]
        confidence = (
            sum(r.confidence for r in usable) / len(usable) if usable else 0.0
        )
        self._decision_trace.append(DecisionRecord(
            action="force_synthesize",
            signals={"usable_results": len(usable)},
            reason=reason,
            timestamp=time.time(),
        ))
        self._memory_store["final_report"] = ResearchReport(
            query=self._query,
            content="\n\n".join(sections) or "No usable evidence was collected before timeout.",
            confidence=round(confidence * 0.7, 2),
            adversarial_status="skipped",
            adversarial_reason=reason,
        )

    async def _verify_report_evidence(self, report: ResearchReport) -> dict[str, Any]:
        if self.evidence_verifier is None:
            return {}
        try:
            verified = await self.evidence_verifier.verify(report, report.sources)
            from ..evidence import ClaimEvidenceEdge
            report.claim_evidence_edges = [
                ClaimEvidenceEdge(
                    claim_id=result.claim.claim_id,
                    source_id=str(evidence.metadata.get("source_id", "")),
                    relation=result.status.value,
                    confidence=result.confidence,
                    reason=result.reason,
                ).to_dict()
                for result in verified
                for evidence in result.evidence
            ]
            summary = self.evidence_verifier.summary(verified)
            summary["unresolved_claims"] = [
                {
                    "claim_id": result.claim.claim_id,
                    "text": result.claim.text,
                    "status": result.status.value,
                    "reason": result.reason,
                    "risk": result.claim.risk,
                    "source_refs": list(result.claim.source_refs),
                    "candidate_sources": [
                        {
                            "url": evidence.source_url,
                            "title": evidence.title,
                            "span": evidence.source_span[:500],
                        }
                        for evidence in result.evidence[:3]
                        if evidence.source_url
                    ],
                }
                for result in verified
                if result.status.value != "supported"
            ]
            report.evidence_verification = summary
            report.open_questions = list(summary["unresolved_claims"])
            if summary.get("total_claims", 0):
                report.confidence = round(
                    report.confidence
                    * (0.5 + 0.5 * summary.get("support_rate", 0.0)),
                    2,
                )
            return summary
        except Exception as exc:
            self._decision_trace.append(DecisionRecord(
                action="verification_failed",
                reason=str(exc),
                timestamp=time.time(),
            ))
            return {}

    def _prepare_evidence_gap_tasks(self, summary: dict[str, Any]) -> list[str]:
        """Replace a broad evidence-gap replan with a small verification DAG."""
        unresolved = summary.get("unresolved_claims", [])
        if not isinstance(unresolved, list):
            return []
        actionable = [item for item in unresolved if isinstance(item, dict) and str(item.get("text", "")).strip()]
        actionable.sort(
            key=lambda item: (
                0 if item.get("status") == "contradicted" else 1,
                0 if item.get("risk") == "high" else 1,
                str(item.get("claim_id", "")),
            )
        )
        limit = min(
            len(actionable),
            max(0, int(self._config.evidence_replan_max_tasks)),
            max(0, int(self._config.max_sub_questions)),
        )
        selected = actionable[:limit]
        if not selected:
            return []

        existing_ids = {result.task_id for result in self._historical_results}
        self._historical_results.extend(
            result
            for result in self._results
            if result.status == AgentStatus.SUCCESS and result.task_id not in existing_ids
        )
        self._results = []
        self._memory_store["unresolved_claims"] = [item["text"] for item in selected]

        dag = DAG()
        task_map: dict[str, SubTask] = {}
        for index, item in enumerate(selected, 1):
            task_id = f"verify_gap_r{self._replan_count + 1}_{index}"
            claim_text = str(item["text"]).strip()
            candidate_sources = item.get("candidate_sources", [])
            source_lines = [
                f"- {source.get('url', '')}: {str(source.get('span', ''))[:240]}"
                for source in candidate_sources
                if isinstance(source, dict) and source.get("url")
            ]
            source_context = "\n".join(source_lines) or "- 当前没有可定位来源，需要做一次精确检索。"
            dag.add_node(task_id)
            task_map[task_id] = SubTask(
                task_id=task_id,
                task_type=TaskType.VERIFY,
                description=(
                    f"claim_id={item.get('claim_id', task_id)}。围绕原始研究问题“{self._query}”，只核验以下结论：{claim_text}。"
                    "优先用 browser 打开下列候选原文，再按需检索官方、论文或一手来源：\n"
                    f"{source_context}\n"
                    "逐项给出数值、单位、日期和可追溯 URL；"
                    "若证据不足，明确标记 unknown，不要扩展到无关主题。"
                ),
                context_keys=["unresolved_claims"],
                expected_type="factual",
                search_hints=[claim_text[:180], "official primary source"],
            )
        self._dag = dag
        self._task_map = task_map
        print(f"[Replan] Evidence gaps converted to targeted verify tasks: {list(task_map)}")
        return list(task_map)

    def _build_failure_reason(self, results: list[AgentResult]) -> str:
        """分析失败原因，生成给 replanner 的描述。"""
        reasons = []
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
        return "; ".join(reasons) if reasons else "Unknown failure"

    def _rebuild_task_map_from_dag(self) -> dict[str, SubTask]:
        """从 DAG 重建 task_map（当缺少原始 SubTask 信息时使用占位符）。

        实际场景中，planner 应返回完整的 SubTask 列表；
        这里作为降级：为 DAG 中每个节点创建默认 SubTask。
        """
        if self._dag is None:
            return {}

        task_map: dict[str, SubTask] = {}
        for node_id in self._dag:
            deps = self._dag.get_dependencies(node_id)
            if node_id not in self._task_map:
                # 新建占位 SubTask
                task_map[node_id] = SubTask(
                    task_id=node_id,
                    task_type=TaskType.SEARCH,
                    description=f"Auto-generated task for {node_id}",
                    dependencies=deps,
                )
            else:
                # 保留已有信息，更新依赖
                old = self._task_map[node_id]
                task_map[node_id] = SubTask(
                    task_id=old.task_id,
                    task_type=old.task_type,
                    description=old.description,
                    dependencies=deps,
                    context_keys=old.context_keys,
                    timeout_seconds=old.timeout_seconds,
                    priority=old.priority,
                    expected_type=old.expected_type,
                    search_hints=old.search_hints,
                )
        return task_map
