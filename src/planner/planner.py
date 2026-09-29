"""
自适应规划器 (Adaptive Planner)

LLM 驱动的规划器，负责将研究问题分解为结构化的子任务 DAG。
核心能力:
  - 初始规划: 生成 3-8 个子问题的 DAG
  - 增量重规划: 保留 confidence≥0.6 的成功结果，仅修改失败子问题
  - 健壮性 JSON 解析: 支持 markdown 代码块、多余换行等噪声
"""
from __future__ import annotations

import json
import re
from typing import Any

from .dag import DAG, DAGCycleError
from .budget_tracker import BudgetTracker
from ..orchestrator.schemas import SubTask, TaskType, AgentResult
from ..utils.tracing import trace_chain
from ..utils.runtime_context import runtime_context_text, resolve_relative_dates


__all__ = ["Planner", "PlanParseError"]


class PlanParseError(Exception):
    """规划结果解析失败时抛出。"""
    pass


# ============================================================================
# Prompt 常量
# ============================================================================

INITIAL_PLAN_PROMPT = """\
You are an expert research planner. Your task is to decompose a complex research question into a directed acyclic graph (DAG) of sub-tasks.

## Input
Research Question: {query}

## Output Format
Return a JSON object with this exact structure (no markdown, no extra text):
{{
  "sub_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "search",
      "description": "What is ...",
      "dependencies": [],
      "context_keys": [],
      "timeout_seconds": 120,
      "priority": 1,
      "expected_type": "factual",
      "search_hints": ["keyword1", "keyword2"]
    }}
  ]
}}

## Rules
1. task_type must be one of: search, analyze, verify
2. dependencies must reference existing task_id values
3. The graph must be a DAG (no cycles)
4. Generate 3 to 6 sub_tasks for normal research; for deep research with no memory generate 5 to 8. Never generate more than 8.
5. More fundamental/information-gathering tasks should have fewer dependencies
6. Verification tasks should depend on analysis tasks
7. Use concise but clear descriptions
8. CRITICAL — RELEVANCE CONSTRAINT: Each sub-task description MUST directly address the research question. If the user asks about 'internship/job application', do NOT generate tasks about 'technology trends', 'annual news summary', or 'science breakthroughs'.
9. The search_hints field MUST contain keywords directly from the query. Do NOT invent unrelated keywords.
10. Prefer specific, actionable queries over broad, vague ones.

## Anti-examples (DO NOT do this)
- Query: "How to find an internship at a big tech company" → BAD tasks: "2025 technology trends", "annual science news", "latest AI breakthroughs"
- Query: "How to prepare for post-training LLM engineer internship" → GOOD tasks: "Big tech post-training intern JD requirements", "LLM post-training intern interview experience", "Resume tips for LLM algorithm intern"

## Runtime context
{runtime_context}

Resolve relative dates using the runtime context. For time-sensitive questions,
include explicit years in search_hints (for example, "2026 LPL Worlds").

## Context (if any)
{memory_context}
"""

REPLAN_PROMPT = """\
You are an expert research planner. Some sub-tasks failed and need to be re-planned.

## Original Question
{query}

## Failed Tasks
{failed_tasks_json}

## Successful Results to Preserve (confidence >= 0.6)
{preserved_results_json}

## Reason for Failure
{reason}

## Output Format
Return a JSON object with new sub_tasks. You may:
1. Modify failed tasks (new task_id, same or different description)
2. Add new tasks to fill gaps
3. Remove tasks that are no longer needed
4. Keep dependencies consistent

Structure:
{{
  "sub_tasks": [...]
}}

Only return the JSON. No markdown, no extra text.
"""


class Planner:
    """自适应规划器。

    Attributes:
        policy: VLLMPolicy 实例，用于调用 LLM。
        budget_tracker: 可选的预算追踪器，监控 planning 阶段的 token 消耗。
    """

    def __init__(
        self,
        policy,
        budget_tracker: BudgetTracker | None = None,
        max_tasks: int = 8,
        max_attempts: int = 3,
        facet_planning_enabled: bool = False,
    ) -> None:
        self.policy = policy
        self.budget_tracker = budget_tracker or BudgetTracker()
        self._last_raw_json: str = ""
        self.max_attempts = max(1, int(max_attempts))
        self.max_tasks = max(1, min(int(max_tasks), 8))
        self.facet_planning_enabled = bool(facet_planning_enabled)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="planner.generate_plan", tags=["m2", "planner"])
    def generate_plan(self, query: str, memory_context: str = "") -> DAG:
        """生成初始执行计划（DAG）。

        Args:
            query: 原始研究问题。
            memory_context: 历史上下文（首次规划为空字符串）。

        Returns:
            DAG: 子任务依赖图。

        Raises:
            PlanParseError: LLM 输出无法解析为合法 DAG 时抛出。
        """
        prompt = self._build_prompt(query, memory_context)
        messages = [
            {"role": "system", "content": f"{runtime_context_text()}\n\nYou are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        return self._generate_with_retry(messages, phase="planning")

    @trace_chain(name="planner.replan", tags=["m2", "planner"])
    def replan(
        self,
        query: str,
        failed_tasks: list[SubTask],
        existing_results: list[AgentResult],
        reason: str,
        preserve_threshold: float = 0.6,
    ) -> DAG:
        """增量重规划：保留高置信度结果，修改失败任务。

        Args:
            query: 原始研究问题。
            failed_tasks: 执行失败的 SubTask 列表。
            existing_results: 所有历史执行结果。
            reason: 失败原因描述。

        Returns:
            DAG: 新的执行计划。
        """
        # 筛选保留的结果（confidence >= 0.6 且状态为 SUCCESS）
        preserved = []
        for r in existing_results:
            if r.status.value != "success" or r.confidence < preserve_threshold:
                continue
            source_urls: list[str] = []
            for step in r.trajectory:
                payload = step.get("result") if isinstance(step, dict) else None
                if not isinstance(payload, dict):
                    continue
                for item in (payload.get("results", []) or payload.get("papers", [])):
                    if isinstance(item, dict):
                        url = item.get("url") or item.get("pdf_url")
                        if url and str(url) not in source_urls:
                            source_urls.append(str(url))
            preserved.append({
                "task_id": r.task_id,
                "output": str(r.output)[:500] if r.output else "",
                "confidence": r.confidence,
                "source_urls": source_urls[:12],
            })

        failed_json = json.dumps(
            [{"task_id": t.task_id, "description": t.description, "type": t.task_type.value} for t in failed_tasks],
            ensure_ascii=False,
            indent=2,
        )
        preserved_json = json.dumps(preserved, ensure_ascii=False, indent=2)

        prompt = REPLAN_PROMPT.format(
            query=resolve_relative_dates(query),
            failed_tasks_json=failed_json,
            preserved_results_json=preserved_json,
            reason=reason,
        )
        if self.facet_planning_enabled:
            prompt += (
                "\nPreserve or add facet_id, completion_criteria, risk_question, and claim_ids on every new sub_task. "
                "Reuse an existing facet_id when the task deepens the same aspect; do not expand the task budget."
            )
        messages = [
            {"role": "system", "content": f"{runtime_context_text()}\n\nYou are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        return self._generate_with_retry(messages, phase="replanning")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, memory: str) -> str:
        """构建初始规划 prompt。"""
        # 首次运行（无历史记忆）时，提示 Planner 更激进地拆解子任务
        has_memory = bool(memory and memory.strip() and memory != "None")
        if not has_memory:
            extra_hint = (
                "\n## Note\n"
                "No previous research memory is available for this topic. "
                "Please be MORE AGGRESSIVE in decomposition: generate 5-8 sub_tasks to thoroughly cover the topic, "
                "rather than the usual 3-6. Never exceed 8 tasks. Each sub-task should focus on a distinct angle or data source.\n"
                "IMPORTANT: Each sub-task description must directly reflect the user's original intent. "
                "If the user asks about 'internship application strategies', do NOT generate tasks about '2025 tech trends' or 'annual science summary'."
            )
        else:
            extra_hint = (
                "\n## Note\n"
                "Use the preserved successful results above to inform new sub-tasks. "
                "New tasks should fill gaps and avoid duplicating existing coverage."
            )
        prompt = INITIAL_PLAN_PROMPT.format(
            query=resolve_relative_dates(query),
            memory_context=memory or "None",
            runtime_context=runtime_context_text(),
        ) + extra_hint
        if self.facet_planning_enabled:
            prompt += (
                "\n## Coverage graph metadata\n"
                "Partition the question into distinct perspectives/facets. Every sub_task must also include: "
                "facet_id (stable snake_case label), completion_criteria (one or more verifiable outcomes), "
                "risk_question (a counterargument, failure mode, or evidence limitation), and claim_ids (possibly empty). "
                "Cover core facts, mechanism, limitations/counterevidence, temporal scope, and stakeholders when relevant. "
                "Do not increase the number of sub_tasks or the search budget.\n"
            )
        return prompt

    def _generate_with_retry(self, messages: list[dict], phase: str) -> DAG:
        """Call the planner a bounded number of times with concise correction."""
        last_error: PlanParseError | None = None
        working = list(messages)
        for attempt in range(self.max_attempts):
            try:
                response = self.policy(working)
                if getattr(response, "get", None) and response.get("status") == "failed":
                    raise PlanParseError(f"LLM call failed during {phase}: {response.get('error', 'unknown error')}")
                content = response.get("content", "") or ""
                finish_reason = response.get("finish_reason") if hasattr(response, "get") else getattr(response, "finish_reason", None)
                self._last_raw_json = content
                self.budget_tracker.track(len(content) // 3)
                return self._parse_plan(content, finish_reason=finish_reason)
            except PlanParseError as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    working = list(messages) + [{
                        "role": "user",
                        "content": (
                            "Your previous output was invalid or truncated. Return ONLY one complete valid JSON object. "
                            f"Keep the plan concise. Generate at most {self.max_tasks} sub_tasks."
                        ),
                    }]
            except Exception as exc:
                last_error = PlanParseError(f"LLM call failed during {phase}: {exc}")
        raise last_error or PlanParseError(f"{phase} failed")

    def _parse_plan(self, json_str: str, finish_reason: str | None = None) -> DAG:
        """健壮性 JSON 解析：处理 markdown 代码块、多余换行等噪声。

        解析策略:
          1. 先尝试直接 json.loads
          2. 失败则提取 markdown 代码块内容
          3. 清理常见噪声（尾部逗号、注释等）
          4. 验证 DAG 无环
        """
        raw = (json_str or "").strip()
        if not raw:
            if finish_reason == "length":
                raise PlanParseError("Planner output was truncated because max_tokens was exhausted.")
            raise PlanParseError("Planner returned empty content.")
        if finish_reason == "length":
            raise PlanParseError("Planner output was truncated because max_tokens was exhausted.")

        # 尝试提取 markdown 代码块
        if raw.startswith("```"):
            # 去掉首行 ```json 或 ```
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # 尝试提取 ```json...``` 中间的内容（即使不在开头）
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # 尝试直接找最外层的 JSON 对象
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()

        # 清理尾部逗号（JSON 不允许 trailing comma）
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            cleaned_lines = []
            for line in raw.splitlines():
                if "//" in line:
                    line = line[: line.index("//")]
                cleaned_lines.append(line)

            cleaned = "\n".join(cleaned_lines)

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e2:
                raise PlanParseError(
                    f"Failed to parse planner output as JSON.\n"
                    f"First error: {e.msg} at line {e.lineno}, column {e.colno}, pos {e.pos}\n"
                    f"Second error: {e2.msg} at line {e2.lineno}, column {e2.colno}, pos {e2.pos}\n"
                    f"Raw output:\n{json_str}"
                ) from e2

        if not isinstance(data, dict) or "sub_tasks" not in data:
            raise PlanParseError(f"Planner output missing 'sub_tasks' key. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        sub_tasks_raw = data["sub_tasks"]
        if not isinstance(sub_tasks_raw, list):
            raise PlanParseError(f"'sub_tasks' must be a list, got {type(sub_tasks_raw)}")
        if not sub_tasks_raw:
            raise PlanParseError("Planner returned an empty sub_tasks list.")
        if len(sub_tasks_raw) > self.max_tasks:
            raise PlanParseError(
                f"Planner returned {len(sub_tasks_raw)} sub_tasks; maximum is {self.max_tasks}."
            )

        dag = DAG()
        task_ids: list[str] = []
        for item in sub_tasks_raw:
            if not isinstance(item, dict):
                raise PlanParseError("Each sub_task must be a JSON object.")
            task = self._deserialize_subtask(item)
            if not task.task_id.strip() or task.task_id == "unknown":
                raise PlanParseError("Every sub_task must have a non-empty task_id.")
            if task.task_id in task_ids:
                raise PlanParseError(f"Duplicate task_id: {task.task_id}")
            task_ids.append(task.task_id)
            dag.add_node(task.task_id)

        # 第二遍添加边
        for item in sub_tasks_raw:
            task_id = item.get("task_id", "")
            for dep in item.get("dependencies", []):
                if not dag.has_node(dep):
                    raise PlanParseError(
                        f"Task '{task_id}' references unknown dependency '{dep}'."
                    )
                dag.add_edge(dep, task_id)  # dep -> task_id (task_id 依赖 dep)

        # 验证无环
        try:
            dag.topological_sort()
        except DAGCycleError as e:
            raise PlanParseError(f"Planner generated a cyclic graph: {e}") from e

        return dag

    def _deserialize_subtask(self, item: dict[str, Any]) -> SubTask:
        """将 JSON dict 反序列化为 SubTask。"""
        def string_list(value: Any) -> list[str]:
            # LLMs occasionally emit a scalar for a schema field declared as
            # an array. ``list("text")`` silently turns it into characters,
            # which polluted Strategy 2 coverage criteria and prompts.
            if isinstance(value, str):
                return [value] if value.strip() else []
            if not isinstance(value, (list, tuple, set)):
                return []
            return [str(entry) for entry in value if str(entry).strip()]

        task_type_str = item.get("task_type", "search")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.SEARCH  # 默认值降级

        return SubTask(
            task_id=item.get("task_id", "unknown"),
            task_type=task_type,
            description=item.get("description", ""),
            dependencies=string_list(item.get("dependencies", [])),
            context_keys=string_list(item.get("context_keys", [])),
            timeout_seconds=int(item.get("timeout_seconds", 120)),
            priority=int(item.get("priority", 1)),
            expected_type=item.get("expected_type", "factual"),
            search_hints=string_list(item.get("search_hints", [])),
            facet_id=str(item.get("facet_id", "") or ""),
            claim_ids=string_list(item.get("claim_ids", [])),
            completion_criteria=string_list(item.get("completion_criteria", [])),
            risk_question=str(item.get("risk_question", "") or ""),
        )

    def get_task_map_from_dag(self, dag: DAG, raw_json: str) -> dict[str, SubTask]:
        """从 DAG 和原始 JSON 重建 task_id -> SubTask 映射。

        通常在 generate_plan 后由编排器调用。
        """
        # 复用 _parse_plan 中的解析逻辑，但返回映射
        # 这里重新解析 raw_json 以获取完整 SubTask 信息
        raw = raw_json.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        sub_tasks_raw = data.get("sub_tasks", [])
        return {item.get("task_id", f"task_{i}"): self._deserialize_subtask(item)
                for i, item in enumerate(sub_tasks_raw)}
