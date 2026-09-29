"""
研究员 Agent (ResearcherAgent)

执行搜索和分析类 SubTask，实现多轮 tool-calling 循环。
设计为项目一 ToolAgentLoop 的简化版：
  - 单 trajectory，无批处理
  - 支持 7 种工具：web_search, arxiv_reader, code_sandbox, browser,
    file_reader, calculator, notepad
  - 通过 VLLMPolicy 进行 LLM 调用
  - 工具结果回写后自动继续，直到模型不再调用工具或达到 max_turns
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, TaskType
from ..tools.execution_policy import ToolExecutionPolicy
from ..tools.search_controller import SearchController
from ..utils.tracing import trace_agent
from ..utils.runtime_context import current_date, runtime_context_text, resolve_relative_dates


__all__ = ["ResearcherAgent"]


class ResearcherAgent(BaseAgent):
    """研究员 Agent：负责搜索、分析、验证类任务。

    可用工具（7 个）：
      - web_search:   网页搜索，返回标题/链接/摘要
      - browser:      网页阅读器，打开 URL 提取正文
      - arxiv_reader: ArXiv 论文元数据检索
      - file_reader:  本地文件阅读（.txt/.md/.pdf/.csv/.json/.docx）
      - code_sandbox: Python 代码沙箱执行
      - calculator:   轻量数学计算（比沙箱更快更安全）
      - notepad:      草稿笔记（记录中间结论/待办/搜索策略）

    Attributes:
        max_turns: 最大交互轮数，防止无限循环。
        tool_map: 工具名称到工具实例的映射。
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 10,
        tool_policy: ToolExecutionPolicy | None = None,
        search_controller: SearchController | None = None,
        evidence_gain_threshold: float = 0.15,
        evidence_patience: int = 1,
        max_tool_calls: int | None = None,
    ) -> None:
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}
        self.tool_policy = tool_policy or ToolExecutionPolicy()
        # AgentPool copies the tools list but intentionally shares the tool
        # instances. Attach one controller to the shared web-search tool so
        # independent workers deduplicate one another's queries and URLs.
        shared_controller = search_controller
        web_tool = self.tool_map.get("web_search")
        if shared_controller is None and web_tool is not None:
            shared_controller = getattr(web_tool, "search_controller", None)
            if shared_controller is None:
                shared_controller = SearchController()
                try:
                    setattr(web_tool, "search_controller", shared_controller)
                except Exception:
                    pass
        self.search_controller = shared_controller or SearchController()
        self.evidence_gain_threshold = min(max(float(evidence_gain_threshold), 0.0), 1.0)
        self.evidence_patience = max(int(evidence_patience), 1)
        self.max_tool_calls = max_tool_calls if max_tool_calls is None else max(int(max_tool_calls), 1)

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行 Researcher 任务。

        流程:
          1. 构建初始 system + user messages
          2. 循环调用 policy，解析 tool_calls
          3. 执行工具，将结果追加为 tool message
          4. 直到无 tool_calls 或达到 max_turns
        """
        trajectory: list[dict] = []
        total_tokens: int = 0
        empty_search_observed = False
        search_budget_stop_observed = False
        successful_search_observed = False
        tool_call_count = 0
        low_novelty_streak = 0
        evidence_seen: set[str] = set()
        self._active_search_context = {
            "query": context.get("query", ""),
            "task_id": task.task_id,
            "stage": "verify" if task.task_type == TaskType.VERIFY else "worker",
            "task_description": task.description,
            "search_hints": list(task.search_hints or []),
            "query_type": task.expected_type,
            "unresolved_claims": context.get("unresolved_claims", []),
            "remaining_search_budget": self.max_tool_calls or self.max_turns,
            "facet_id": task.facet_id,
            "facet": context.get("facet", task.description),
            "claim_ids": list(task.claim_ids or []),
            "source_cluster_ids": context.get("source_cluster_ids", []),
            "action": context.get("frontier_action", "search_new_facet"),
            "estimated_value": context.get("frontier_estimated_value"),
        }

        # 构建任务描述
        task_desc = self._build_task_prompt(task, context)

        # 查询可行性判断：如果任务明显无法通过网络搜索获得答案，直接基于已知信息分析
        if self._is_non_searchable(task, context):
            messages = [
                {"role": "system", "content": self._system_prompt_direct_analysis()},
                {"role": "user", "content": task_desc},
            ]
            try:
                response = self.policy(messages)
                if getattr(response, "get", None) and response.get("status") == "failed":
                    raise RuntimeError(response.get("error", "LLM backend unavailable"))
                content = response.get("content", "") or ""
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=[{"role": "assistant", "content": content}],
                    token_usage=len(content) // 3,
                    confidence=self._extract_confidence(content),
                )
            except Exception as e:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Direct analysis failed: {e}",
                    trajectory=[{"error": str(e)}],
                    token_usage=0,
                    confidence=0.0,
                )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task_desc},
        ]

        # 若 policy 支持 tool 设置，则注册可用工具
        if hasattr(self.policy, "set_tools") and self.tools:
            schemas = [t.get_openai_tool_schema() for t in self.tools]
            self.policy.set_tools(schemas)

        # 根据任务类型确定 fallback 工具
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        fallback_tool = "arxiv_reader" if any(kw in desc_lower for kw in academic_keywords) else "web_search"
        
        for turn in range(self.max_turns):
            # Fallback: if last turn had no tool_calls, force a search instruction
            if turn > 0 and messages and messages[-1].get("role") == "assistant":
                last_tool_calls = messages[-1].get("tool_calls", [])
                if not last_tool_calls:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You did not use any tools. "
                            f"You MUST call the '{fallback_tool}' tool now to search for information. "
                            f"Do not write a summary without searching first."
                        ),
                    })

            try:
                # 使用线程池执行同步 policy，避免阻塞 asyncio 事件循环
                response = await asyncio.to_thread(self.policy, messages)
            except RuntimeError as e:
                # 上下文长度超限等致命错误
                trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=str(e),
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=0.0,
                )

            if getattr(response, "get", None) and response.get("status") == "failed":
                error = response.get("error", "LLM backend unavailable")
                trajectory.append({"turn": turn, "error": error})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"LLM call failed: {error}",
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=0.0,
                )

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []

            trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
            })

            # 估算 token（简化：字符数 / 3）
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            # 无工具调用 → 任务完成
            if not tool_calls:
                # Factual/searchable tasks must use at least one tool. The
                # previous implementation accepted a turn-zero answer from
                # model memory, which violated the system prompt and yielded
                # unsupported claims.
                if turn == 0:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You did not use any tools. You MUST call the '{fallback_tool}' tool now "
                            "before answering. Return the final summary only after using the tool."
                        ),
                    })
                    trajectory.append({"turn": turn, "event": "tool_required", "fallback_tool": fallback_tool})
                    continue
                # B方案：检测 LLM 回复是否包含明显的工具失败说明
                if self._is_tool_failure_explanation(content):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=content,
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )
                dependency_evidence_available = any(
                    key.startswith("dep:")
                    and isinstance(value, AgentResult)
                    and value.status == AgentStatus.SUCCESS
                    and bool(value.output)
                    for key, value in context.items()
                )
                if (
                    empty_search_observed
                    and not successful_search_observed
                    and not (search_budget_stop_observed and dependency_evidence_available)
                ):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=content or "Search returned no usable results.",
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )
                confidence = self._extract_confidence(content)
                if task.task_type == TaskType.VERIFY:
                    content = self._normalise_verify_output(content, task)
                final_result = AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=confidence,
                )
                from ..evidence import EvidenceLedger
                final_result.evidence_bundle = EvidenceLedger().collect([final_result]).to_dict()
                return final_result

            # 执行工具调用
            tool_results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "web_search":
                    args = self._normalize_temporal_search_args(args, task, context)
                tool_call_count += 1
                self._active_search_context["remaining_search_budget"] = max(
                    int(self.max_tool_calls or self.max_turns) - tool_call_count, 0
                )
                if self.max_tool_calls is not None and tool_call_count > self.max_tool_calls:
                    result = {
                        "error_type": "budget",
                        "hard_cap_reached": True,
                        "stop_search_requested": True,
                        "message": "tool call budget exhausted; summarize gathered evidence or state what remains unknown",
                    }
                    search_budget_stop_observed = True
                else:
                    result = await self._execute_tool(tool_name, args, context=context)

                # B方案：检测工具返回结果是否包含 error 字段
                if (
                    isinstance(result, dict)
                    and result.get("error")
                    and result.get("error_type") != "budget"
                ):
                    error_msg = result["error"]
                    trajectory.append({
                        "turn": turn,
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "error": error_msg,
                    })
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=f"Tool '{tool_name}' failed: {error_msg}",
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "args": dict(args),
                    "result": result,
                })
                trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "args": dict(args),
                    "result": result,
                })
                execution_meta = result.get("_tool_execution") if isinstance(result, dict) else None
                if execution_meta:
                    trajectory[-1]["execution"] = execution_meta

            # Evidence-gain stopping: a round with no new evidence is a
            # stronger signal than a fixed search-round count. SearchController
            # reports novelty; the local fallback keeps custom tools compatible.
            all_empty = True
            evidence_action_observed = False
            round_novelty_values: list[float] = []
            for tr in tool_results:
                tool_name = tr["name"]
                if tool_name not in {
                    "web_search", "arxiv_reader", "browser", "file_reader",
                    "calculator", "code_sandbox",
                }:
                    continue
                evidence_action_observed = True
                res = tr["result"]
                if isinstance(res, dict) and res.get("hard_cap_reached"):
                    search_budget_stop_observed = True
                if tool_name == "web_search" and isinstance(res, dict):
                    novelty = res.get("evidence_novelty")
                    if isinstance(novelty, (int, float)):
                        round_novelty_values.append(float(novelty))
                keys = self._evidence_keys_for_tool(tool_name, res)
                if keys:
                    all_empty = False
                    successful_search_observed = True
                    new_keys = keys - evidence_seen
                    evidence_seen.update(keys)
                    if not round_novelty_values or tool_name != "web_search":
                        round_novelty_values.append(len(new_keys) / max(len(keys), 1))
                elif tool_name in {"web_search", "arxiv_reader", "browser", "file_reader"}:
                    empty_search_observed = True

            if round_novelty_values:
                round_novelty = max(round_novelty_values)
            elif all_empty and evidence_action_observed:
                round_novelty = 0.0
            else:
                # Administrative tools such as notepad neither prove evidence
                # gain nor justify an early stop.
                round_novelty = 1.0 if tool_results else 0.0
            if evidence_action_observed and round_novelty <= self.evidence_gain_threshold:
                low_novelty_streak += 1
            elif evidence_action_observed:
                low_novelty_streak = 0
            policy_stop = any(
                isinstance(item.get("result"), dict) and item["result"].get("stop_search_requested")
                for item in tool_results
            )
            force_summary = bool(tool_results) and (
                low_novelty_streak >= self.evidence_patience or policy_stop
            )
            if force_summary:
                trajectory.append({
                    "turn": turn,
                    "event": "policy_stop" if policy_stop else "evidence_gain_stop",
                    "novelty": round(round_novelty, 4),
                    "streak": low_novelty_streak,
                    "unique_evidence": len(evidence_seen),
                })

            # 将 assistant message 和 tool results 追加到 messages
            assistant_msg = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [dict(tc) for tc in tool_calls]
            # 保留 reasoning_content（DeepSeek 推理模型需要传回）
            if response.get("reasoning_content"):
                assistant_msg["reasoning_content"] = response["reasoning_content"]
            messages.append(assistant_msg)

            for tr in tool_results:
                msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
                # 如果强制总结，给工具结果附加提示
                if force_summary:
                    msg_content += "\n\n[SYSTEM NOTICE] You have already searched enough. Write your final summary NOW. Do NOT call any more tools."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": msg_content,
                })

        # 达到 max_turns
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.TIMEOUT,
            output="Reached max_turns without final answer.",
            trajectory=trajectory,
            token_usage=total_tokens,
            confidence=0.0,
        )

    def _system_prompt(self) -> str:
        return (
            runtime_context_text() + "\n\nYou are a meticulous research assistant. "
            "Your job is to gather and analyze information using the RIGHT tool for each task. "
            "\n\nAVAILABLE TOOLS:\n"
            "- web_search: General web search for news, market data, industry reports, current events. "
            "  Use this as the FIRST tool for most tasks.\n"
            "- arxiv_reader: Academic paper search (ArXiv / Semantic Scholar). "
            "  USE when the task involves: papers, publications, academic research, citation counts.\n"
            "- browser: Open a URL and extract full webpage text. "
            "  USE after web_search when search results are too short and you need to read the original article in depth.\n"
            "- code_sandbox: Execute Python code for calculations, data processing, simulations. "
            "  USE when the task requires: computing FLOPs, memory usage, statistical analysis, data transformation.\n"
            "- calculator: Quick math evaluation (+, -, *, /, sqrt, log, mean, etc.). "
            "  USE for simple calculations instead of code_sandbox.\n"
            "- notepad: Write/read intermediate notes to avoid forgetting findings during multi-step research. "
            "  USE to record key numbers, conclusions, or next search queries.\n"
            "- file_reader: Read local files (.txt, .md, .pdf, .csv, .json, .docx). "
            "  USE only when the task explicitly references a local file path.\n"
            "\nIMPORTANT RULES:\n"
            "1. You MUST use a tool to find factual information. Do NOT answer from your own knowledge.\n"
            "2. Choose the RIGHT tool based on the task type. You can use MULTIPLE tools in sequence.\n"
            "3. For most research tasks, START with web_search or arxiv_reader.\n"
            "4. If search results are too short, use browser to read the full article.\n"
            "5. If the task involves numbers/calculations, use calculator or code_sandbox.\n"
            f"6. You may call tools AT MOST {self._tool_call_limit()} times total. After that you MUST summarize.\n"
            "7. Only after gathering information, provide a concise summary with a confidence score (0-1).\n"
            "8. NEVER greet the user or ask what they want to search — just execute immediately.\n"
            f"9. If you have already performed {self._tool_call_limit()} tool calls, do NOT call more — write the final summary now."
        )

    def _system_prompt_direct_analysis(self) -> str:
        return (
            runtime_context_text() + "\n\nYou are a thoughtful analyst. "
            "The user has asked a question that cannot be answered by web search "
            "(e.g., analyzing a specific private individual, personal advice, or subjective judgment). "
            "Your job is to provide a reasoned analysis based ONLY on the information already provided in the context. "
            "Do NOT make up facts. Clearly state what is known, what can be reasonably inferred, and what remains unknown. "
            "End with a confidence score (0-1)."
        )

    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """启发式判断任务是否无法通过网络搜索获取答案。"""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        # 模式 1：分析/评价特定私人个体（姓名 + 描述性分析）
        if "朋友" in combined or "同学" in combined or "同事" in combined:
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        # 模式 2：主观建议类（基于个人情况）
        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):
            if "朋友" in combined or "我" in query:
                return True

        # 模式 3：明显的个人隐私分析
        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        """根据 SubTask 和全局上下文构建 user prompt。"""
        desc_lower = (task.description or "").lower()
        
        # 智能工具推荐：根据任务描述关键词匹配
        tool_recommendations = []
        
        # 学术论文类
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        if any(kw in desc_lower for kw in academic_keywords):
            tool_recommendations.append("arxiv_reader")
        
        # 计算/数学类
        calc_keywords = ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "公式", "推导"]
        if any(kw in desc_lower for kw in calc_keywords):
            tool_recommendations.append("calculator")
            tool_recommendations.append("code_sandbox")
        
        # 深度阅读类（需要读原文）
        browser_keywords = ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]
        if any(kw in desc_lower for kw in browser_keywords):
            tool_recommendations.append("browser")
        
        # 文件类
        file_keywords = ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]
        if any(kw in desc_lower for kw in file_keywords):
            tool_recommendations.append("file_reader")
        
        # 确定首选工具：学术论文类优先用 arxiv_reader，其他先用 web_search
        is_academic = "arxiv_reader" in tool_recommendations
        if is_academic:
            # 学术论文任务：arxiv_reader 优先，web_search 备选
            tool_recommendations = ["arxiv_reader"] + [t for t in tool_recommendations if t != "arxiv_reader"]
        elif not tool_recommendations:
            tool_recommendations.insert(0, "web_search")
        
        primary_tool = tool_recommendations[0]
        secondary_tools = tool_recommendations[1:]
        
        lines = [
            f"## Task: {resolve_relative_dates(task.description)}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
            "",
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(tool_recommendations)}",
        ]
        
        if secondary_tools:
            lines.append(f"Start with '{primary_tool}'. If the task involves numbers/calculations, also use {', '.join(secondary_tools)}.")
        else:
            lines.append(f"Use '{primary_tool}' to gather information.")
        
        lines.extend([
            "",
            "## INSTRUCTIONS:",
            f"1. First, call the '{primary_tool}' tool with a relevant query to gather information.",
            "2. Review the results.",
            f"3. If needed, continue with a refined query or another appropriate tool within the shared budget.",
            f"   You may call tools AT MOST {self._tool_call_limit()} times total. After the final allowed call, you MUST write the final summary.",
            "4. If search results are too short, you may use 'browser' to read the full article (counts as 1 tool call).",
            "5. If calculations are needed, use 'calculator' or 'code_sandbox' (counts as 1 tool call).",
            "6. Finally, summarize your findings in Chinese with a confidence score (0-1).",
            "7. DO NOT greet the user or ask clarifying questions — just execute immediately.",
            "8. IMPORTANT: Your query MUST directly address the task description.",
        ])
        if task.search_hints:
            lines.insert(1, f"Search hints (MUST use these as primary keywords): {', '.join(task.search_hints)}")
        if task.context_keys:
            ctx_parts = []
            for key in task.context_keys:
                if key in context:
                    ctx_parts.append(f"- {key}: {context[key]}")
            if ctx_parts:
                lines.append("\n## Context:")
                lines.extend(ctx_parts)
        dependency_parts = []
        for dependency_id in task.dependencies:
            dependency = context.get(f"dep:{dependency_id}")
            if isinstance(dependency, AgentResult) and dependency.status == AgentStatus.SUCCESS:
                dependency_parts.append(f"- {dependency_id}: {dependency.output}")
        if dependency_parts:
            lines.append("\n## Dependency evidence (use before requesting more searches):")
            lines.extend(dependency_parts)
        if context.get("compressed_context"):
            lines.append("\n## Compressed dependency context:")
            lines.append(str(context["compressed_context"]))
        if task.task_type == TaskType.VERIFY:
            lines.extend([
                "\n## REQUIRED VERIFY OUTPUT CONTRACT:",
                "Return JSON only: {\"claim_id\": string, \"status\": \"supported\"|\"contradicted\"|\"unknown\", "
                "\"evidence\": [{\"url\": string, \"span\": string}], \"reason\": string, \"confidence\": number}.",
                "Never return supported without a non-empty attributable evidence span. When uncertain, return unknown.",
            ])
        return "\n".join(lines)

    @staticmethod
    def _normalise_verify_output(content: str, task: SubTask) -> str:
        """Fail closed while keeping every VERIFY result machine-readable."""
        claim_match = re.search(r"claim_id=([\w:-]+)", task.description)
        claim_id = claim_match.group(1) if claim_match else task.task_id
        raw = content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            payload = {
                "claim_id": claim_id,
                "status": "unknown",
                "evidence": [],
                "reason": "unparseable_verify_output",
                "confidence": 0.0,
                "raw_summary": content[:1000],
            }
        if not isinstance(payload, dict):
            payload = {"claim_id": claim_id, "status": "unknown", "evidence": [], "reason": "invalid_verify_shape", "confidence": 0.0}
        payload["claim_id"] = str(payload.get("claim_id") or claim_id)
        if payload.get("status") not in {"supported", "contradicted", "unknown"}:
            payload["status"] = "unknown"
        evidence = payload.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence = [
            item for item in evidence
            if isinstance(item, dict) and str(item.get("url", "")).strip() and str(item.get("span", "")).strip()
        ]
        payload["evidence"] = evidence
        if payload["status"] == "supported" and not evidence:
            payload["status"] = "unknown"
            payload["reason"] = "supported_without_attributable_span"
        try:
            payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        except (TypeError, ValueError):
            payload["confidence"] = 0.0
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _tool_call_limit(self) -> int:
        """Return the single budget value rendered into every prompt."""
        return int(self.max_tool_calls or self.max_turns)

    @staticmethod
    def _evidence_keys_for_tool(tool_name: str, result: Any) -> set[str]:
        """Extract stable evidence identities from every evidence-producing tool.

        The controller may provide a more precise novelty score for web
        search.  This fallback prevents arXiv/browser/file/calculation calls
        from being mistaken for empty searches.
        """
        if isinstance(result, dict) and result.get("error_type") == "budget":
            return set()
        values: list[Any] = []
        if isinstance(result, dict):
            for field in ("results", "papers", "sources", "evidence"):
                candidate = result.get(field)
                if isinstance(candidate, list):
                    values.extend(candidate)
            if not values and any(
                key in result for key in ("content", "text", "value", "result", "output")
            ):
                values.append(result)
        elif isinstance(result, str):
            if result.strip() and not result.lstrip().startswith(("[Browser Error]", "[Browser Warning]")):
                values.append(result)
        elif result is not None:
            values.append(result)

        keys: set[str] = set()
        for item in values:
            if isinstance(item, dict):
                url = str(item.get("canonical_url") or item.get("url") or item.get("pdf_url") or "").strip()
                content = str(
                    item.get("source_span") or item.get("content") or item.get("text")
                    or item.get("snippet") or item.get("summary") or item.get("value")
                    or item.get("result") or item.get("output") or item.get("title") or ""
                ).strip()
                identity = url or content
            else:
                identity = str(item).strip()
            if identity:
                keys.add(hashlib.sha256(identity.encode("utf-8")).hexdigest())
        return keys

    async def _execute_tool(self, tool_name: str, args: dict, context: dict | None = None) -> dict:
        """调用具体工具实例。"""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            fallback_tools = self._fallback_tools(tool_name, args)
            if tool_name == "web_search" and self.search_controller is not None:
                return await self.search_controller.execute(
                    tool,
                    args,
                    execution_policy=self.tool_policy,
                    fallback_tools=fallback_tools,
                    context=getattr(self, "_active_search_context", None) or context,
                )
            return await self.tool_policy.execute(
                tool_name,
                tool,
                args,
                fallback_tools=fallback_tools,
            )
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _fallback_tools(self, tool_name: str, args: dict) -> list[tuple[str, Any, dict]]:
        """Build schema-compatible provider fallbacks when available."""
        if tool_name == "web_search" and "arxiv_reader" in self.tool_map:
            return [
                (
                    "arxiv_reader",
                    self.tool_map["arxiv_reader"],
                    {"query": args.get("query", ""), "max_results": args.get("top_n", 3)},
                )
            ]
        if tool_name == "arxiv_reader" and "web_search" in self.tool_map:
            return [
                (
                    "web_search",
                    self.tool_map["web_search"],
                    {"query": args.get("query", ""), "top_n": args.get("max_results", 5)},
                )
            ]
        return []

    @staticmethod
    def _normalize_temporal_search_args(args: dict, task: SubTask, context: dict) -> dict:
        """Bias relative-date searches toward the current year.

        Historical queries remain untouched; for queries containing relative
        expressions, append the explicit current year when it is missing.
        """
        normalized = dict(args)
        query = str(normalized.get("query", "")).strip()
        source_text = f"{context.get('query', '')} {task.description}".lower()
        relative = ("今年", "本年", "当前", "目前", "最新", "近期", "最近", "this year", "current", "latest", "recent")
        if query and any(token in source_text for token in relative):
            year = current_date()[:4]
            if not re.search(rf"\b{re.escape(year)}\b", query):
                query = f"{query} {year}"
            normalized["query"] = query
        return normalized

    def _is_tool_failure_explanation(self, content: str) -> bool:
        """检测 LLM 回复是否是工具失败的解释说明而非真实研究结果。

        常见模式：额度用完、无法连接、无法搜索等。
        """
        if not content:
            return False
        c = content.lower()
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "配额已用完", "额度已用完", "搜索配额",
            "cannot search", "unable to search", "quota exceeded",
            "api key", "额度不足", "余额不足", "余额为", "余额：0",
            "网络错误", "连接失败", "无法连接到",
        ]
        return any(kw in c for kw in failure_keywords)

    def _extract_confidence(self, content: str) -> float:
        """从输出文本中尝试提取置信度分数。"""
        import re
        # 匹配 "Confidence: 0.85" 或 "置信度: 0.85"
        patterns = [
            r"[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)",
            r"置信度[:\s]+(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        # 默认中等置信度
        return 0.6
