"""
合成 Agent (SummarizerAgent)

将多个 SubTask 的执行结果合成为结构化的研究报告。
区别于 ResearcherAgent 的多轮 tool-calling，Summarizer 是单轮长上下文生成任务：
  - 把所有子结果按置信度排序后拼接为上下文
  - 调用 LLM 一次性生成 Markdown 格式报告
  - 提取引用来源，计算整体置信度
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.tracing import trace_agent
from ..utils.runtime_context import runtime_context_text
from ..utils.temporal import infer_source_date
from ..evidence import EvidenceLedger


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """合成 Agent：将子任务结果合成为最终研究报告。

    Attributes:
        max_output_tokens: 报告生成的最大 token 数（通过 policy.max_tokens 控制）。
    """

    def __init__(self, name: str, policy, tools: list | None = None) -> None:
        super().__init__(name, policy, tools)

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行合成任务。

        Args:
            task: 通常是一个特殊的 "synthesize" 类型任务。
            context: 全局上下文，必须包含 "results" 和 "query" 键。
                results: list[AgentResult]
                query: str 原始研究问题

        Returns:
            AgentResult，output 字段为 ResearchReport 实例。
        """
        query = context.get("query", "")
        results: list[AgentResult] = context.get("results", [])

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # 构建 synthesis prompt
        source_catalog = context.get("source_catalog") or self._collect_sources(query, results)
        prompt = self._build_synthesis_prompt(
            query, results, source_catalog, prior_report=context.get("prior_report")
        )
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # 合成任务不需要工具调用，临时禁用 tools 避免模型进入 tool-calling 模式
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy(messages)
        except RuntimeError as e:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=str(e),
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            )
        finally:
            if "old_tools" in locals():
                self.policy.tools = old_tools

        if getattr(response, "get", None) and response.get("status") == "failed":
            error = response.get("error", "LLM backend unavailable")
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=error,
                trajectory=[{"error": error}],
                token_usage=0,
                confidence=0.0,
            )

        content = response.get("content", "") or ""
        token_usage = len(content) // 3  # 简化估算

        # 解析报告内容，提取来源和置信度
        report = self._parse_report(query, content, results, source_catalog)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

    def _system_prompt(self) -> str:
        return (
            runtime_context_text() + " "
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly. "
            "Write only as much as the available evidence supports; never pad the report to meet a length target. "
            "Write in depth where evidence permits: include background, key findings, analysis, comparisons, and implications. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "For time-sensitive questions, distinguish current evidence from historical/stale evidence; "
            "never use an older source as a current fact without an explicit caveat. "
            "At the end, provide an overall confidence score (0-1); the runtime appends the canonical source list."
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        results: list[AgentResult],
        source_catalog: list[dict] | None = None,
        prior_report: ResearchReport | None = None,
    ) -> str:
        """构建合成 prompt，按置信度降序排列结果。"""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        parts = [
            runtime_context_text() + "\n",
            f"# Research Question\n{query}\n",
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{r.output}\n"
            )

        if prior_report is not None and prior_report.content:
            parts.append(
                "\n# Prior Verified Draft\n"
                "Preserve supported sections verbatim where possible. Revise only claims addressed by new verification evidence.\n"
                f"{prior_report.content}\n"
            )

        source_catalog = source_catalog if source_catalog is not None else self._collect_sources(query, results)
        parts.append(f"\n# Source Catalog ({len(source_catalog)} sources)\n")
        for source in source_catalog:
            parts.append(
                f"[{source['citation_id']}] {source.get('title', '')}\n"
                f"Stable source ID: {source.get('source_id', '')}\n"
                f"URL: {source.get('url', '')}\n"
                f"Evidence span: {str(source.get('source_span') or source.get('snippet', ''))[:800]}\n"
            )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report based on the findings above. Do NOT say 'I will synthesize'.\n"
            "2. Use a refer-then-claim loop for each factual sentence: first select the exact evidence span and its catalog ID, "
            "then write one atomic sentence supported by that span and append [N] immediately.\n"
            "3. Never introduce a new factual detail while paraphrasing. Separate analysis/inference from sourced facts.\n"
            "4. Structure only the sections supported by the evidence; do not pad to a minimum length.\n"
            "5. Resolve contradictions explicitly. Every number, date, comparison, causal claim, and externally verifiable fact "
            "MUST cite matching Source Catalog IDs using [N]. Never invent an ID.\n"
            "6. Do not write a separate references/source list; the runtime appends the canonical catalog.\n"
            "7. If the catalog lacks support, state that the point is unknown or omit it.\n"
            "8. End with: Overall Confidence: X.XX"
        )
        return "\n".join(parts)

    def _parse_report(
        self,
        query: str,
        content: str,
        results: list[AgentResult],
        source_catalog: list[dict] | None = None,
    ) -> ResearchReport:
        """从 LLM 输出中解析 ResearchReport，并基于子任务成功率校准置信度。"""
        # 1. 从文本中提取 LLM 自评置信度
        llm_confidence = 0.5
        m = re.search(r"[Oo]verall\s+[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)", content)
        if m:
            try:
                llm_confidence = float(m.group(1))
            except ValueError:
                pass

        # 2. 基于子任务成功率计算客观置信度
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. 综合置信度 = LLM 自评 × 成功率开根（降低成功率的影响权重）
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        unique_sources = source_catalog if source_catalog is not None else self._collect_sources(query, results)

        # Do not present stale evidence as a high-confidence current answer.
        is_temporal_query = any(token in query.lower() for token in ("今年", "当前", "目前", "最新", "近期", "最近", "this year", "current", "latest", "recent"))
        if is_temporal_query and not unique_sources:
            # A current answer with no extracted evidence must not retain the
            # model's optimistic self-reported confidence.
            confidence *= 0.5
            confidence = round(max(0.0, min(1.0, confidence)), 2)
        elif unique_sources and is_temporal_query:
            relevance = [s.get("temporal_relevance") for s in unique_sources]
            if "current" not in relevance:
                confidence *= 0.5
            elif any(v == "historical_context" for v in relevance):
                confidence *= 0.85
            confidence = round(max(0.0, min(1.0, confidence)), 2)

        # 统计实际工具调用次数（遍历所有子任务的 trajectory）
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
            evidence=list(unique_sources),
            source_catalog=list(unique_sources),
        )

    def _collect_sources(self, query: str, results: list[AgentResult]) -> list[dict]:
        """Build a bounded, stable and source-diverse citation catalog."""
        successful = [result for result in results if result.status == AgentStatus.SUCCESS]
        max_sources = min(40, max(12, len(successful) * 4))
        sources = EvidenceLedger(max_sources_per_task=12).catalog(successful)[:max_sources]
        for source in sources:
            source_date = infer_source_date(source)
            source["source_date"] = source_date
            source["temporal_relevance"] = self._temporal_relevance(query, source_date)
        return sources

    @staticmethod
    def _temporal_relevance(query: str, source_date: str | None) -> str:
        if not any(token in query.lower() for token in ("今年", "当前", "目前", "最新", "近期", "最近", "this year", "current", "latest", "recent")):
            return "not_applicable"
        if not source_date:
            return "unknown"
        try:
            source_year = int(str(source_date)[:4])
            current_year = datetime.now().astimezone().year
        except (TypeError, ValueError):
            return "unknown"
        return "current" if source_year == current_year else ("historical_context" if source_year < current_year else "future_dated")
