"""
M5 Blue Agent — 修复与防御器

Blue Agent 接收 Red Agent 的 Verdict，按优先级排序并执行三类修复：
1. In-place Fix：数字/日期与 source 不一致 → 直接替换
2. Supplementary Search：unsourced claims → 触发新搜索
3. Removal：高置信幻觉 → 删除段落

修复后执行 self_verify，确保不引入新矛盾。
"""
from __future__ import annotations

import asyncio
import copy
import difflib
import inspect
import re
from dataclasses import dataclass
from typing import Any

from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent
from src.utils.runtime_context import runtime_context_text


__all__ = ["BlueAgent", "IssueContext"]


@dataclass(frozen=True)
class IssueContext:
    """A bounded report view centered on one Red issue."""

    excerpt: str
    target_text: str
    target_start: int
    target_end: int
    matched_by: str
    confidence: float
    source_ids: tuple[int, ...] = ()


# ============================================================================
# Prompt 模板
# ============================================================================

SYSTEM_BLUE_AGENT = (
    runtime_context_text() + "\n\n你是一位严谨的研究报告修订员（Blue Agent）。你的任务是根据审查意见修复研究报告，"
    "确保所有修改都有据可依，不引入新错误。输出必须是 JSON 格式。"
)

PROMPT_SELF_VERIFY = """请验证以下【本次局部变更】是否引入了新的矛盾、事实错误、语病或引用错误。
只判断本次变更产生的问题，不要因报告中原本存在、但未被本次修改的内容而拒绝修复。

请按以下 JSON 格式输出：
{
  "has_new_issue": bool,
  "new_issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string"
    }
  ]
}

--- 变更前局部上下文 ---
{original}

--- 变更后局部上下文 ---
{revised}

--- 已执行的修复 ---
{fixes}
"""

PROMPT_IN_PLACE_FIX = """请根据以下审查意见，对研究报告进行【原地修正】。

要求：
1. 仅修改与 source 不一致的具体数字、日期、名字等事实性内容。
2. 保持原文结构和叙述风格不变。
3. 所有修改必须基于提供的 sources，不能引入新信息。
4. fixed_content 必须为空，只通过 changes 返回精确补丁，禁止重写整篇报告。
5. 只能修改 <TARGET>...</TARGET> 内的内容；报告开头和结尾仅用于理解上下文。

请按以下 JSON 格式输出：
{
  "changes": [                 // 变更记录列表
    {
      "location": "string",
      "before": "string",
      "after": "string"
    }
  ],
  "fixed_content": ""         // 必须为空
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 来源 ---
{sources}
"""

PROMPT_SUPPLEMENTARY_SEARCH = """请根据以下审查意见，对研究报告进行【补充搜索后修正】。

要求：
1. 对无 source 支撑的 claim，使用搜索结果补充证据。
2. 如果搜索结果无法证实，则删除该 claim 或标注为"未经证实"。
3. 只修改与本条审查意见直接相关的句子，不重写无关段落。
4. changes 中必须提供报告里可精确匹配的 before 原文和替换后的 after；长报告将依赖该补丁安全合并。
5. fixed_content 必须为空，禁止输出或续写整篇报告。
6. 搜索结果只能用 {{SOURCE_1}} 这类占位符引用；禁止自行编造 [21] 等数字引用。程序会把占位符映射为正式引用。
7. 若搜索摘要不足以支持 claim，应删除、弱化或标注未经证实，不得强行引用。
8. 只能修改 <TARGET>...</TARGET> 内的内容。

请按以下 JSON 格式输出：
{
  "changes": [
    {
      "location": "string",
      "action": "added|removed|modified",
      "before": "string",
      "after": "string",
      "detail": "string"
    }
  ],
  "fixed_content": ""  // 必须为空
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}

--- 搜索结果 ---
{search_results}
"""

PROMPT_REMOVAL = """请根据以下审查意见，对研究报告进行【移除修正】。

要求：
1. 改写包含幻觉 claim 的完整句子，而不是只删除句中片段。
2. after 必须是语法完整、上下文连贯的替换句；不得产生残缺标点或空句。
3. 只修改与审查意见直接相关的最小完整句子。
4. fixed_content 必须为空，通过 changes 返回可精确应用的补丁。
5. 只能修改 <TARGET>...</TARGET> 内的内容。

请按以下 JSON 格式输出：
{
  "changes": [
    {
      "location": "string",
      "before": "报告中的完整原句",
      "after": "移除错误信息后的完整替换句",
      "detail": "string"
    }
  ],
  "fixed_content": ""  // 必须为空
}

--- 审查意见 ---
{issue_desc}

--- 原始报告 ---
{content}
"""


# ============================================================================
# Blue Agent 实现
# ============================================================================

class BlueAgent:
    """Blue Agent — 修复与防御器。

    Attributes:
        policy: VLLMPolicy 实例。
        tools: 可用工具列表，至少包含搜索工具用于 supplementary_search。
        max_tokens: 单次修复调用的最大输出 token。
    """

    def __init__(
        self,
        policy,
        tools: list[Any] | None = None,
        max_tokens: int = 4096,
        max_issues: int = 5,
        max_sources: int = 20,
        max_consecutive_failures: int = 2,
        max_candidate_sources: int = 3,
        repair_context_chars: int = 4000,
        self_verify_context_chars: int = 6000,
        search_controller: Any | None = None,
        search_execution_policy: Any | None = None,
    ):
        self.policy = policy
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.max_issues = max(1, int(max_issues))
        self.max_sources = max(1, int(max_sources))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.max_candidate_sources = max(1, int(max_candidate_sources))
        self.repair_context_chars = max(1000, min(int(repair_context_chars), 32000))
        self.self_verify_context_chars = max(
            1000, min(int(self_verify_context_chars), 32000)
        )
        # 缓存搜索工具
        self._search_tool = self._find_search_tool()
        self.search_controller = search_controller or getattr(self._search_tool, "search_controller", None)
        self.search_execution_policy = search_execution_policy
        self.status = "ready"
        self.error = ""
        self.last_repair_stats: dict[str, int] = {}

    def _find_search_tool(self) -> Any | None:
        """从 tools 列表中查找搜索工具。"""
        for t in self.tools:
            name = getattr(t, "name", "")
            if "search" in name.lower():
                return t
        return None

    @trace_agent(name="blue_agent.defend", tags=["m5", "blue", "adversarial"])
    async def defend(
        self, report: ResearchReport, verdict: RedVerdict
    ) -> tuple[ResearchReport, list[FixOperation]]:
        """根据 Red Verdict 修复研究报告。

        执行流程：
        1. 按优先级对 issues 排序。
        2. 逐个执行修复（in_place / search / removal）。
        3. 每轮修复后执行 self_verify，检测是否引入新问题。
        4. 返回修复后的报告和所有 FixOperation 记录。

        Args:
            report: 原始研究报告（不会被修改，内部深拷贝）。
            verdict: Red Agent 的审查结果。

        Returns:
            (fixed_report, fix_operations)
        """
        current = copy.deepcopy(report)
        operations: list[FixOperation] = []
        self.status = "success"
        self.error = ""
        self.last_repair_stats = {
            "selected": 0,
            "attempted": 0,
            "committed": 0,
            "rolled_back": 0,
            "skipped_after_failure_cap": 0,
        }

        if not verdict.issues:
            return current, operations

        # Red 正常会先截断；这里再次去重和封顶，避免外部构造的 Verdict
        # 绕过预算边界。
        sorted_issues = self._select_issues(verdict.issues)
        committed = 0
        attempted = 0
        failed = 0
        consecutive_failures = 0
        errors: list[str] = []

        for issue in sorted_issues:
            attempted += 1
            before = copy.deepcopy(current)
            try:
                op = await self._fix_single_issue(current, issue)
            except Exception as exc:
                current = before
                self.status = "partial" if committed else "failed"
                self.error = str(exc)
                operations.append(FixOperation(issue=issue, action="backend_failed", success=False, detail=str(exc)))
                failed += 1
                errors.append(str(exc))
                # Backend/transport exceptions are unlikely to recover within
                # the same request, so do not amplify a failing dependency.
                break
            operations.append(op)
            if not op.success:
                current = before
                failed += 1
                consecutive_failures += 1
                errors.append(op.detail or op.action)
                if consecutive_failures >= self.max_consecutive_failures:
                    break
                continue

            # 每个修复都是一个小事务：只有相对上一份已验证报告通过
            # self_verify 后才提交；失败只回滚当前修复，不丢弃此前提交。
            try:
                verify_pass, verify_issues = await self._self_verify(
                    before.content, current.content, operations
                )
            except Exception as exc:
                verify_pass = False
                verify_issues = [Issue(
                    severity=Severity.MAJOR,
                    dimension=Dimension.LOGICAL,
                    description=f"Blue Agent self-verification failed: {exc}",
                    fix_type=FixType.IN_PLACE,
                )]
                self.error = str(exc)
            if not verify_pass:
                current = before
                op.success = False
                op.detail = f"{op.detail}; rolled_back_after_self_verify"
                for vi in verify_issues:
                    operations.append(
                        FixOperation(
                            issue=vi,
                            action="self_verify_detected_new_issue",
                            success=False,
                            detail=vi.description,
                        )
                    )
                self.status = "partial" if committed else "failed"
                failed += 1
                consecutive_failures += 1
                errors.append(self.error or "self_verify_detected_new_issue")
                if consecutive_failures >= self.max_consecutive_failures:
                    break
                continue
            committed += 1
            consecutive_failures = 0

        skipped = len(sorted_issues) - attempted
        self.last_repair_stats = {
            "selected": len(sorted_issues),
            "attempted": attempted,
            "committed": committed,
            "rolled_back": failed,
            "skipped_after_failure_cap": skipped,
        }
        if failed:
            self.status = "partial" if committed else "failed"
            self.error = "; ".join(error for error in errors[-3:] if error)
        else:
            self.status = "success"
            self.error = ""

        return current, operations

    def _select_issues(self, issues: list[Issue]) -> list[Issue]:
        """Defensively deduplicate and cap repair operations."""
        unique: list[Issue] = []
        seen: set[tuple[str, str, str, str]] = set()
        for issue in issues:
            dimension = getattr(issue.dimension, "value", str(issue.dimension))
            fix_type = getattr(issue.fix_type, "value", str(issue.fix_type))
            location = " ".join((issue.location or "").lower().split())
            description = " ".join((issue.description or "").lower().split())
            key = (dimension, location, fix_type, description if not location else "")
            if key in seen:
                continue
            seen.add(key)
            unique.append(issue)
        unique.sort(key=VerdictEngine.compute_priority, reverse=True)
        return unique[: self.max_issues]

    @staticmethod
    def _require_response(resp):
        if getattr(resp, "get", None) and resp.get("status") == "failed":
            raise RuntimeError(resp.get("error", "Blue Agent backend unavailable"))
        return resp

    async def _call_policy(self, messages: list[dict[str, str]]) -> Any:
        """Call sync policies off-loop so the enclosing hard timeout can fire."""
        call = self.policy
        call_method = getattr(call, "__call__", None)
        if inspect.iscoroutinefunction(call) or inspect.iscoroutinefunction(call_method):
            result = call(messages)
        else:
            result = await asyncio.to_thread(call, messages)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _fix_single_issue(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """对单个 Issue 执行修复。"""
        old_max = getattr(self.policy, "max_tokens", None)
        if old_max is not None:
            self.policy.max_tokens = self.max_tokens

        try:
            if issue.fix_type == FixType.IN_PLACE:
                result = await self._do_in_place_fix(report, issue)
            elif issue.fix_type == FixType.SUPPLEMENTARY:
                result = await self._do_supplementary_search(report, issue)
            elif issue.fix_type == FixType.REMOVAL:
                result = await self._do_removal(report, issue)
            else:
                result = FixOperation(
                    issue=issue,
                    action="unknown_fix_type",
                    success=False,
                    detail=f"未知的 fix_type: {issue.fix_type}",
                )
        finally:
            if old_max is not None:
                self.policy.max_tokens = old_max

        return result

    async def _do_in_place_fix(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行原地修正。"""
        context = self._build_issue_context(report, issue)
        prompt = PROMPT_IN_PLACE_FIX
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", context.excerpt)
        prompt = prompt.replace(
            "{sources}", self._format_sources(
                report.sources,
                max_items=self.max_sources,
                priority_ids=context.source_ids,
            )
        )
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self._require_response(await self._call_policy(messages))
        raw = (resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")) or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content or changes:
            # The prompt may contain a head/tail excerpt for a long report.
            # Never replace the complete report with a model response that
            # silently omitted the unseen middle/suffix.
            merged = self._merge_fixed_content(
                report.content,
                fixed_content,
                changes,
                target_range=(context.target_start, context.target_end),
            )
            if merged == report.content:
                return FixOperation(
                    issue=issue,
                    action="in_place_fix_rejected",
                    success=False,
                    detail=(
                        "candidate did not contain an applicable patch inside target; "
                        f"context={context.matched_by}:{context.confidence:.2f}"
                    ),
                )
            report.content = merged
            return FixOperation(
                issue=issue,
                action=f"in_place_fix: {changes}",
                success=True,
                detail=(
                    f"context={context.matched_by}:{context.confidence:.2f}, "
                    f"changes={changes}"
                ),
            )
        return FixOperation(
            issue=issue,
            action="in_place_fix_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_supplementary_search(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行补充搜索后修正。"""
        context = self._build_issue_context(report, issue)
        search_results = ""
        candidates: list[dict[str, str]] = []
        search_failed = False
        if self._search_tool is not None:
            try:
                # 假设搜索工具有 async execute 或同步 execute 接口
                query = issue.description
                if self.search_controller is not None:
                    sr = await self.search_controller.execute(
                        self._search_tool,
                        {"query": query},
                        execution_policy=self.search_execution_policy,
                        context={"stage": "blue", "task_id": "adversarial_supplementary_search"},
                    )
                elif hasattr(self._search_tool, "execute"):
                    if hasattr(self._search_tool.execute, "__call__"):
                        if inspect.iscoroutinefunction(self._search_tool.execute):
                            sr = await self._search_tool.execute(query=query)
                        else:
                            sr = self._search_tool.execute(query=query)
                            if inspect.isawaitable(sr):
                                sr = await sr
                    else:
                        sr = None
                else:
                    sr = None
                candidates = self._normalize_search_results(sr)
                search_results = self._format_search_candidates(candidates)
                search_failed = not candidates
            except Exception as e:
                search_results = f"（搜索失败: {e}）"
                search_failed = True
        else:
            search_results = "（无可用搜索工具）"
            search_failed = True

        if search_failed:
            return FixOperation(
                issue=issue,
                action="supplementary_search_failed",
                success=False,
                detail=search_results[:500],
            )

        prompt = PROMPT_SUPPLEMENTARY_SEARCH
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", context.excerpt)
        prompt = prompt.replace("{search_results}", search_results)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self._require_response(await self._call_policy(messages))
        raw = (resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")) or ""

        _fixed_content, changes = self._parse_fix_json(raw)
        if changes:
            prepared, additions, reference_lines, validation_error = (
                self._prepare_supplementary_changes(report, changes, candidates)
            )
            if validation_error:
                return FixOperation(
                    issue=issue,
                    action="supplementary_search_rejected",
                    success=False,
                    detail=(
                        f"{validation_error}; "
                        f"context={context.matched_by}:{context.confidence:.2f}"
                    ),
                )
            merged = self._merge_fixed_content(
                report.content,
                "",
                prepared,
                target_range=(context.target_start, context.target_end),
            )
            if merged == report.content:
                return FixOperation(
                    issue=issue,
                    action="supplementary_search_rejected",
                    success=False,
                    detail=(
                        "candidate did not contain an applicable patch inside target; "
                        f"context={context.matched_by}:{context.confidence:.2f}"
                    ),
                )
            if reference_lines:
                heading = "" if "### 补充来源" in merged else "\n\n### 补充来源"
                merged = merged.rstrip() + heading + "\n" + "\n".join(reference_lines) + "\n"
            report.content = merged
            report.sources.extend(additions)
            return FixOperation(
                issue=issue,
                action=f"supplementary_search: {prepared}",
                success=True,
                detail=(
                    f"context={context.matched_by}:{context.confidence:.2f}, "
                    f"candidate_sources={len(candidates)}, "
                    f"added_sources={len(additions)}, changes={prepared}"
                ),
            )
        return FixOperation(
            issue=issue,
            action="supplementary_search_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_removal(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """执行移除修正。"""
        context = self._build_issue_context(report, issue)
        prompt = PROMPT_REMOVAL
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", context.excerpt)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self._require_response(await self._call_policy(messages))
        raw = (resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")) or ""

        _fixed_content, changes = self._parse_fix_json(raw)
        if changes:
            merged = self._merge_fixed_content(
                report.content,
                "",
                changes,
                target_range=(context.target_start, context.target_end),
            )
            if merged == report.content:
                return FixOperation(
                    issue=issue,
                    action="removal_rejected",
                    success=False,
                    detail=(
                        "candidate did not contain an applicable patch inside target; "
                        f"context={context.matched_by}:{context.confidence:.2f}"
                    ),
                )
            report.content = merged
            return FixOperation(
                issue=issue,
                action=f"removal_rewrite: {changes}",
                success=True,
                detail=(
                    f"context={context.matched_by}:{context.confidence:.2f}, "
                    f"changes={changes}"
                ),
            )
        return FixOperation(
            issue=issue,
            action="removal_failed",
            success=False,
            detail=raw[:500],
        )

    async def _self_verify(
        self, original: str, revised: str, operations: list[FixOperation]
    ) -> tuple[bool, list[Issue]]:
        """修复后自验证，检查是否引入新矛盾。

        Returns:
            (是否通过, 新发现的 issues 列表)
        """
        if not revised or revised == original:
            return True, []

        fixes_text = "\n".join(
            f"- [{op.issue.dimension.value}] {op.action}: {op.detail[:200]}"
            for op in operations[-5:]  # 只取最近5条，避免 prompt 过长
        )
        before_context, after_context = self._change_contexts(
            original, revised, max_len=self.self_verify_context_chars
        )
        prompt = PROMPT_SELF_VERIFY
        prompt = prompt.replace("{original}", before_context)
        prompt = prompt.replace("{revised}", after_context)
        prompt = prompt.replace("{fixes}", fixes_text)
        messages = [
            {"role": "system", "content": SYSTEM_BLUE_AGENT},
            {"role": "user", "content": prompt},
        ]
        resp = self._require_response(await self._call_policy(messages))
        raw = (resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")) or ""

        try:
            data = self._parse_json_dict(raw)
            has_new = bool(data.get("has_new_issue", False))
            new_issues = []
            for item in data.get("new_issues", []):
                try:
                    dimension = Dimension(item.get("dimension", "logical"))
                except (ValueError, TypeError):
                    dimension = Dimension.LOGICAL
                try:
                    severity = Severity(item.get("severity", "minor"))
                except (ValueError, TypeError):
                    severity = Severity.MINOR
                new_issues.append(
                    Issue(
                        severity=severity,
                        dimension=dimension,
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        fix_type=FixType.IN_PLACE,
                    )
                )
            return not has_new, new_issues
        except Exception as exc:
            # An unavailable/invalid verifier is a failed defense, never a
            # successful verification fallback.
            self.error = str(exc)
            return False, [Issue(
                severity=Severity.MAJOR,
                dimension=Dimension.LOGICAL,
                description=f"Blue Agent self-verification failed: {exc}",
                fix_type=FixType.IN_PLACE,
            )]

    @staticmethod
    def _parse_json_dict(raw: str) -> dict[str, Any]:
        """Parse a JSON object while tolerating the common Markdown wrapper."""
        import json
        import re

        text = (raw or "").strip()
        if not text:
            raise ValueError("empty JSON response")
        candidates = [text]
        candidates.extend(
            match.strip()
            for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        )
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if not isinstance(value, dict):
                    raise ValueError("JSON response must be an object")
                return value
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"invalid JSON object: {last_error}")

    def _normalize_search_results(self, payload: Any) -> list[dict[str, str]]:
        """Convert tool-specific search output into bounded, citeable records."""
        raw_results: Any = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_results, list):
            return []
        normalized: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            snippet = str(item.get("snippet", item.get("content", "")) or "").strip()
            if not url or not title or not snippet or url in seen_urls:
                continue
            seen_urls.add(url)
            normalized.append({"title": title, "url": url, "snippet": snippet[:500]})
            if len(normalized) >= self.max_candidate_sources:
                break
        return normalized

    @staticmethod
    def _format_search_candidates(candidates: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"{{{{SOURCE_{index}}}}}\nTitle: {source['title']}\n"
            f"URL: {source['url']}\nSnippet: {source['snippet']}"
            for index, source in enumerate(candidates, 1)
        )

    def _prepare_supplementary_changes(
        self,
        report: ResearchReport,
        changes: list[dict],
        candidates: list[dict[str, str]],
    ) -> tuple[list[dict], list[dict[str, str]], list[str], str]:
        """Bind model placeholders to deterministic report citation IDs.

        Numeric citation IDs authored directly by the model are accepted only
        when that ID already exists in the report. Newly searched sources are
        added exclusively through SOURCE_n placeholders.
        """
        existing_ids = {int(value) for value in re.findall(r"\[(\d+)\]", report.content)}
        next_id = max(existing_ids | {len(report.sources)}, default=0) + 1
        existing_urls = {
            str(source.get("url", "") or ""): index
            for index, source in enumerate(report.sources, 1)
            if isinstance(source, dict) and source.get("url")
        }
        additions: list[dict[str, str]] = []
        reference_lines: list[str] = []
        placeholder_ids: dict[int, int] = {}
        prepared: list[dict] = []

        for raw_change in changes:
            if not isinstance(raw_change, dict):
                continue
            change = dict(raw_change)
            before = str(change.get("before", "") or "")
            after = str(change.get("after", "") or "")
            if not before or not after:
                continue

            direct_ids = {int(value) for value in re.findall(r"\[(\d+)\]", after)}
            invented = direct_ids - existing_ids
            if invented:
                return [], [], [], (
                    "model introduced unbound numeric citations: "
                    + ", ".join(f"[{value}]" for value in sorted(invented))
                )

            # Models commonly wrap the requested placeholder in another pair
            # of brackets. Match that whole form so ``[{{SOURCE_1}}]`` becomes
            # ``[21]`` rather than the malformed ``[[21]]``.
            placeholder_pattern = re.compile(
                r"\[\s*\{\{SOURCE_(\d+)\}\}\s*\]"
                r"|\{\{SOURCE_(\d+)\}\}"
                r"|\[SOURCE_(\d+)\]"
            )
            matches = list(placeholder_pattern.finditer(after))
            for match in matches:
                candidate_index = int(next(group for group in match.groups() if group))
                if candidate_index < 1 or candidate_index > len(candidates):
                    return [], [], [], f"unknown source placeholder: SOURCE_{candidate_index}"
                if candidate_index not in placeholder_ids:
                    source = candidates[candidate_index - 1]
                    url = source["url"]
                    citation_id = existing_urls.get(url)
                    if citation_id is None:
                        citation_id = next_id
                        next_id += 1
                        placeholder_ids[candidate_index] = citation_id
                        additions.append(dict(source))
                        title = (
                            source["title"].replace("\n", " ")
                            .replace("[", "").replace("]", "").strip()
                        )
                        snippet = source["snippet"].replace("\n", " ").strip()
                        reference_lines.append(
                            f"[{citation_id}] [{title}]({url}) — {snippet}"
                        )
                    else:
                        placeholder_ids[candidate_index] = citation_id
                after = after.replace(match.group(0), f"[{placeholder_ids[candidate_index]}]")
            change["before"] = before
            change["after"] = after
            prepared.append(change)

        if not prepared:
            return [], [], [], "no complete before/after patch was returned"
        return prepared, additions, reference_lines, ""

    @staticmethod
    def _change_contexts(original: str, revised: str, max_len: int = 4000) -> tuple[str, str]:
        """Return bounded contexts around actual edits, excluding unrelated text."""
        matcher = difflib.SequenceMatcher(a=original, b=revised, autojunk=False)
        before_parts: list[str] = []
        after_parts: list[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            before_parts.append(original[max(0, i1 - 250):min(len(original), i2 + 250)])
            after_parts.append(revised[max(0, j1 - 250):min(len(revised), j2 + 250)])
        before = "\n\n[下一处变更]\n\n".join(before_parts) or original
        after = "\n\n[下一处变更]\n\n".join(after_parts) or revised
        before = before[:max_len]
        if "### 补充来源" in revised:
            source_section = revised[revised.rfind("### 补充来源"):]
            if source_section not in after[:max_len]:
                source_section = source_section[:1200]
                prefix_len = max(0, max_len - len(source_section) - 2)
                after = after[:prefix_len] + "\n\n" + source_section
        return before, after[:max_len]

    @staticmethod
    def _normalize_locator(value: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", (value or "").casefold())

    @staticmethod
    def _paragraph_bounds(content: str, position: int) -> tuple[int, int]:
        start = content.rfind("\n\n", 0, position)
        start = 0 if start < 0 else start + 2
        end = content.find("\n\n", position)
        return start, len(content) if end < 0 else end

    def _locate_issue(self, content: str, issue: Issue) -> tuple[int, int, int, str, float]:
        """Locate the narrowest auditable report region for one issue."""
        metadata = re.search(r"(?m)^##\s+(?:元信息|参考来源|References)\s*$", content)
        body_end = metadata.start() if metadata else len(content)
        body = content[:body_end]
        quoted: list[str] = []
        for value in (issue.description, issue.evidence):
            quoted.extend(
                match[0] or match[1]
                for match in re.findall(
                    r"“([^”]{8,})”|\"([^\"\n]{8,})\"",
                    value or "",
                )
            )
        if issue.evidence and len(issue.evidence.strip()) >= 12:
            quoted.append(issue.evidence.strip())
        for needle in sorted(set(quoted), key=len, reverse=True):
            position = body.find(needle)
            if position >= 0:
                start, end = self._paragraph_bounds(content, position)
                return start, min(end, body_end), position, "exact_claim", 1.0

        location_parts = [
            part.strip(" #*-—:：")
            for part in re.split(r"[>/|]", issue.location or "")
            if len(self._normalize_locator(part)) >= 2
        ]
        headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", body))
        best_heading: tuple[int, re.Match[str]] | None = None
        raw_location = issue.location or ""
        location_sections = set(
            re.findall(r"(?<!\d)(\d+(?:\.\d+)+)", raw_location)
        )
        for heading in headings:
            heading_norm = self._normalize_locator(heading.group(2))
            heading_sections = set(
                re.findall(r"(?<!\d)(\d+(?:\.\d+)+)", heading.group(2))
            )
            if location_sections and not location_sections.intersection(heading_sections):
                continue
            score = 1000 if location_sections.intersection(heading_sections) else 0
            for part in location_parts or [raw_location]:
                part_norm = self._normalize_locator(part)
                if not part_norm or not heading_norm:
                    continue
                if part_norm in heading_norm or heading_norm in part_norm:
                    score = max(score, 100 + min(len(part_norm), len(heading_norm)))
                    continue
                common = difflib.SequenceMatcher(
                    None, part_norm, heading_norm, autojunk=False
                ).find_longest_match().size
                if common >= 4:
                    score = max(score, common)
            if score and (best_heading is None or score > best_heading[0]):
                best_heading = (score, heading)
        if best_heading is not None:
            heading = best_heading[1]
            level = len(heading.group(1))
            end = body_end
            for following in headings:
                if following.start() > heading.start() and len(following.group(1)) <= level:
                    end = following.start()
                    break
            return heading.start(), end, heading.end(), "heading", 0.85

        issue_text = " ".join(
            part for part in (issue.location, issue.description, issue.evidence) if part
        )
        citation_ids = re.findall(r"\[(\d+)\]", issue_text)
        for citation_id in citation_ids:
            match = re.search(rf"\[{re.escape(citation_id)}\]", body)
            if match:
                start, end = self._paragraph_bounds(content, match.start())
                return start, min(end, body_end), match.start(), "citation", 0.7

        needle = self._normalize_locator(issue_text)
        best_fuzzy: tuple[int, int, int] | None = None
        if needle:
            for paragraph in re.finditer(r"(?s)(?:^|\n\s*\n)(.*?)(?=\n\s*\n|$)", body):
                paragraph_text = paragraph.group(1)
                candidate = self._normalize_locator(paragraph_text)
                if not candidate:
                    continue
                longest = difflib.SequenceMatcher(
                    None, needle, candidate, autojunk=False
                ).find_longest_match().size
                if longest >= 8 and (best_fuzzy is None or longest > best_fuzzy[0]):
                    start = paragraph.start(1)
                    best_fuzzy = (longest, start, paragraph.end(1))
        if best_fuzzy is not None:
            longest, start, end = best_fuzzy
            confidence = min(0.65, 0.4 + longest / 100)
            return start, end, start, "fuzzy_claim", confidence
        return 0, body_end, 0, "head_tail_fallback", 0.0

    def _build_issue_context(
        self,
        report: ResearchReport,
        issue: Issue,
        max_chars: int | None = None,
    ) -> IssueContext:
        """Build a bounded head + target + tail view for a single repair."""
        content = report.content
        budget = max(1000, int(max_chars or self.repair_context_chars))
        start, end, anchor, matched_by, confidence = self._locate_issue(content, issue)
        source_ids = tuple(sorted({
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                " ".join((issue.location, issue.description, issue.evidence, content[start:end])),
            )
            if 1 <= int(value) <= len(report.sources)
        }))
        open_tag = "<TARGET>\n"
        close_tag = "\n</TARGET>"

        if matched_by == "head_tail_fallback":
            target_text = content[start:end]
            target = self._excerpt(
                target_text, max(1, budget - len(open_tag) - len(close_tag))
            )
            return IssueContext(
                open_tag + target + close_tag,
                target_text,
                start,
                end,
                matched_by,
                confidence,
                source_ids,
            )

        head_label = "[报告开头，仅供上下文]\n"
        before_label = "\n\n[目标前文，仅供上下文]\n"
        target_label = "\n\n[当前问题目标区域]\n"
        after_label = "\n\n[目标后文，仅供上下文]\n"
        tail_label = "\n\n[报告结尾，仅供上下文]\n"
        global_cap = max(100, min(600, (budget - 300) // 4))
        local_cap = max(80, min(400, (budget - 300) // 6))
        head = content[:global_cap] if start > global_cap + 100 else ""
        tail = content[-global_cap:] if end < len(content) - global_cap - 100 else ""
        before_local = content[max(0, start - local_cap):start]
        after_local = content[end:min(len(content), end + local_cap)]
        overhead = (
            (len(head_label) if head else 0)
            + (len(before_label) if before_local else 0)
            + len(target_label) + len(open_tag) + len(close_tag)
            + (len(after_label) if after_local else 0)
            + (len(tail_label) if tail else 0)
        )
        target_budget = max(
            200,
            budget - overhead - len(head) - len(tail)
            - len(before_local) - len(after_local),
        )
        target_start, target_end = start, end
        if end - start > target_budget:
            relative_anchor = max(start, min(anchor, end))
            target_start = max(start, relative_anchor - target_budget // 2)
            target_end = min(end, target_start + target_budget)
            target_start = max(start, target_end - target_budget)
        target = content[target_start:target_end]
        excerpt = (
            (head_label + head if head else "")
            + (before_label + before_local if before_local else "")
            + target_label + open_tag + target + close_tag
            + (after_label + after_local if after_local else "")
            + (tail_label + tail if tail else "")
        )
        return IssueContext(
            excerpt,
            target,
            target_start,
            target_end,
            matched_by,
            confidence,
            source_ids,
        )

    def _format_sources(
        self,
        sources: list[dict],
        max_items: int = 15,
        priority_ids: tuple[int, ...] = (),
    ) -> str:
        """格式化来源列表，截断以避免上下文膨胀。"""
        if not sources:
            return "（无来源）"
        lines = []
        valid_priority = [value for value in priority_ids if 1 <= value <= len(sources)]
        ordered_ids = valid_priority + [
            value for value in range(1, len(sources) + 1) if value not in valid_priority
        ]
        for i in ordered_ids[:max_items]:
            s = sources[i - 1]
            title = s.get("title", "未知标题")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]
            lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... 还有 {len(sources) - max_items} 个来源未显示")
        return "\n".join(lines)

    @staticmethod
    def _excerpt(content: str, max_len: int = 4000) -> str:
        """Return a bounded head/tail excerpt without implying it is complete.

        A head-only excerpt was previously fed back to the model as if it were
        the complete report.  Keeping both ends gives the model enough context
        for citations and makes it possible to validate that a candidate still
        contains the report's suffix before accepting it.
        """
        if len(content) <= max_len:
            return content
        marker = "\n\n[报告中间内容省略；请仅返回明确修改，勿删除未显示内容]\n\n"
        available = max(0, max_len - len(marker))
        head_len = available // 2
        tail_len = available - head_len
        return content[:head_len] + marker + content[-tail_len:]

    def _truncate_content(self, content: str, max_len: int | None = None) -> str:
        """Compatibility alias used by repair prompts."""
        return self._excerpt(content, max_len or self.repair_context_chars)

    def _merge_fixed_content(
        self,
        original: str,
        candidate: str,
        changes: list[dict] | None,
        *,
        removal: bool = False,
        target_range: tuple[int, int] | None = None,
    ) -> str:
        """Apply an LLM repair while preserving text outside its excerpt.

        Blue's repair prompts ask for full content, but long reports are sent
        as bounded excerpts.  If a model returns a shortened report, replacing
        the original would lose everything after the excerpt.  Explicit patch
        records are applied to the original first; a candidate is accepted as
        a whole only when it retains both the original head and tail (or the
        report was short enough to be shown in full).
        """
        operations = changes if isinstance(changes, list) else []
        patched = original
        patch_count = 0
        target_start, target_end = target_range or (0, len(original))
        target_start = max(0, min(target_start, len(original)))
        target_end = max(target_start, min(target_end, len(original)))
        allowed_original = original[target_start:target_end]
        for item in operations:
            if not isinstance(item, dict):
                continue
            if removal:
                before = str(item.get("original_text", item.get("before", "")) or "")
                if (
                    before
                    and before in allowed_original
                    and original.count(before) == 1
                    and before in patched
                ):
                    patched = patched.replace(before, "", 1)
                    patch_count += 1
                continue
            before = str(item.get("before", "") or "")
            after = str(item.get("after", "") or "")
            if (
                before
                and before in allowed_original
                and original.count(before) == 1
                and before in patched
                and after
            ):
                patched = patched.replace(before, after, 1)
                patch_count += 1

        # Prefer exact patches over a model-authored whole-report rewrite.
        # This is both cheaper and substantially easier to audit.
        if patch_count:
            return patched

        candidate = (candidate or "").strip()
        if not candidate:
            return original
        if target_range is not None and target_range != (0, len(original)):
            return original
        # Whole-report candidates are a compatibility fallback only. Even when
        # the complete input fitted in the prompt, require both anchors so a
        # larger context budget cannot silently widen the truncation surface.
        anchor_len = max(1, min(600, len(original) // 4))
        has_head = original[:anchor_len] in candidate
        has_tail = original[-anchor_len:] in candidate
        if has_head and has_tail:
            return candidate.replace("[报告中间内容省略；请仅返回明确修改，勿删除未显示内容]", "").strip()
        # Without anchors or an explicit patch, fail closed.  A near-complete
        # candidate can still silently omit the final paragraph, which is much
        # worse than asking the next Red/Blue round to retry the repair.
        return original

    def _parse_fix_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 in_place / search 修复的 JSON 输出。"""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("changes", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                pass
        return "", []

    def _parse_removal_json(self, raw: str) -> tuple[str, list[dict]]:
        """解析 removal 修复的 JSON 输出。"""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("removed_segments", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                pass
        return "", []
