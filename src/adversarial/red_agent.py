"""
M5 Red Agent — 五维度对抗攻击器

Red Agent 的职责是对研究报告进行多维度"攻击"，找出事实错误、幻觉、逻辑矛盾、
来源可信度问题和覆盖缺失。每个维度有独立的 Prompt 模板，确保评估的细致与全面。

设计决策：
1. 每个维度独立调用 LLM，避免单条 Prompt 过长导致模型注意力分散。
2. 输出强制要求结构化 JSON，降低解析失败率。
3. 对解析失败或后端失败的维度显式标记失败，禁止用默认分数伪装成功。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any

from src.adversarial.verdict import (
    Dimension,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent
from src.utils.runtime_context import runtime_context_text


__all__ = ["RedAgent"]


# ============================================================================
# Prompt 模板 — 每个维度独立、详细、可运行
# ============================================================================

SYSTEM_RED_AGENT = (
    runtime_context_text() + "\n\n你是一位极其严苛的研究报告审查员（Red Agent）。你的任务是以批判性思维深度审查研究报告，"
    "找出所有事实错误、幻觉、逻辑漏洞、来源缺陷和覆盖缺失。你必须基于客观证据给出评分，"
    "不能因报告写作流畅而放松标准。评分标准要严格——多数研究报告默认只有 5-6 分而非 8-9 分。"
    "输出必须是严格的 JSON 格式。"
)

# --- 维度 1: 事实核查 ---
PROMPT_FACTUAL = """请对以下研究报告进行【事实核查】评分。

评分标准（0-10分）：
- 10分：所有可验证的事实（数字、日期、人名、机构名、统计数据）均有可靠来源支撑，且与来源完全一致。
- 7-9分：个别非核心事实缺少直接来源，或存在轻微数值偏差（<5%）。
- 4-6分：存在明显事实错误（日期错误、数据引用错误）但核心论点仍成立。
- 1-3分：多处核心事实错误，严重损害报告可信度。
- 0分：大量事实完全错误，报告基本不可信。

审查要求：
1. 逐条提取报告中的 factual claims（数字、日期、比例、排名等）。
2. 将每条 claim 与提供的 sources 进行比对。
3. 标记不一致或无法验证的 claim。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": float,           // 0-10
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",   // 具体问题描述
      "location": "string",      // 问题位置，如"第3段"或引用标记
      "fix_type": "in_place|search|removal",
      "evidence": "string"       // 支撑证据或 source 原文
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""

# --- 维度 2: 幻觉检测 ---
PROMPT_HALLUCINATION = """请对以下研究报告进行【幻觉检测】评分。

评分标准（0-10分）：
- 10分：报告中的每一条信息都能在 sources 中找到明确支撑，无 hallucination。
- 7-9分：存在少量"合理的推断"但未明确标注为推断，可能误导读者。
- 4-6分：存在明显的无来源陈述，尤其是具体数字、事件细节或因果关系。
- 1-3分：大量段落包含无来源信息，部分信息疑似模型编造。
- 0分：报告充斥着模型幻觉，几乎无可信内容。

审查要求：
1. 逐段检查是否存在无 sources 支撑的 claim。
2. 特别关注：具体数字、精确日期、直接引语、因果关系、排名顺序。
3. 区分"合理推断"与"无依据断言"：推断应有明确标注。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""

# --- 维度 3: 逻辑一致性 ---
PROMPT_LOGICAL = """请对以下研究报告进行【逻辑一致性】评分。

评分标准（0-10分）：
- 10分：论证链条完整，前提与结论一致，无矛盾陈述。
- 7-9分：个别推断稍显跳跃，但不影响整体结论。
- 4-6分：存在内部矛盾（如前文说A，后文说非A）或因果谬误。
- 1-3分：多处逻辑断裂、自相矛盾，核心论点无法自洽。
- 0分：报告逻辑混乱，论证完全不可信。

审查要求：
1. 检查是否存在前后矛盾的陈述。
2. 检查因果关系是否合理（避免 post hoc / 因果倒置）。
3. 检查样本推断总体是否存在以偏概全。
4. 检查比较类论述的基准是否一致。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}
"""

# --- 维度 4: 来源可信度 ---
PROMPT_SOURCE_CREDIBILITY = """请对以下研究报告的【来源可信度】评分。

评分标准（0-10分）：
- 10分：所有来源均为高权威的一手资料（政府官网、顶级期刊、官方财报），且时效性强。
- 7-9分：以权威二手资料为主，个别来源时效稍旧但非核心数据。
- 4-6分：混有低权威来源（匿名论坛、未验证自媒体）且未做交叉验证。
- 1-3分：主要依赖低质量来源，或存在来源循环引用。
- 0分：无来源或来源完全不可信。

审查要求：
1. 评估每个 source 的域名权威性（.gov / .edu / 顶级媒体 / 自媒体 / 未知）。
2. 评估内容类型（一手数据 / 分析报道 / 社论 / 用户生成内容）。
3. 评估时效性：对于快速变化领域（科技、股市），1年以上为陈旧。
4. 检查一手程度：优先一手数据，二手分析需标注原始来源。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- 研究报告 ---
Query: {query}

Content:
{content}

--- 来源列表 ---
{sources}
"""

# --- 维度 5: 覆盖完整度 ---
PROMPT_COVERAGE = """请对以下研究报告的【覆盖完整度】评分。

评分标准（0-10分）：
- 10分：完全覆盖 query 要求的所有子话题，无重要遗漏，正反方观点均衡呈现，且每个子话题的讨论都基于相关搜索结果。
- 7-9分：覆盖了主要子话题，个别边缘视角缺失，但不影响核心结论。搜索结果与查询基本相关。
- 4-6分：遗漏了 query 隐含的关键子话题，或只呈现单方面观点。部分搜索结果可能与查询无关。
- 1-3分：严重跑题（例如搜索内容与查询主题无关）或大量子话题未覆盖。
- 0分：完全未回答 query。

审查要求：
1. 将 query 拆解为应覆盖的子话题列表。
2. 逐一检查每个子话题是否在报告中得到充分讨论。
3. 检查是否存在明显的立场偏差（只呈现正方而忽略反方）。
4. 检查时间维度是否覆盖（历史背景、现状、未来趋势，视 query 需求而定）。
5. CRITICAL: 检查报告中的 sources（搜索来源）是否与 query 主题相关。如果 sources 全是与 query 无关的网页（如搜"实习"却返回"科技趋势"），必须标记为 major/critical issue，并说明搜索内容与查询意图不匹配。

请按以下 JSON 格式输出（不要有任何额外文字）：
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- 原始问题 ---
{query}

--- 研究报告 ---
{content}
"""

# 维度 → Prompt 映射
DIMENSION_PROMPTS: dict[Dimension, str] = {
    Dimension.FACTUAL: PROMPT_FACTUAL,
    Dimension.HALLUCINATION: PROMPT_HALLUCINATION,
    Dimension.LOGICAL: PROMPT_LOGICAL,
    Dimension.SOURCE_CREDIBILITY: PROMPT_SOURCE_CREDIBILITY,
    Dimension.COVERAGE: PROMPT_COVERAGE,
}


# ============================================================================
# Red Agent 实现
# ============================================================================

class RedAgent:
    """Red Agent — 五维度对抗攻击器。

    Attributes:
        policy: VLLMPolicy 实例，提供 LLM 调用能力。
        max_tokens: 单维度评估的最大输出 token 数。
    """

    def __init__(
        self,
        policy,
        max_tokens: int = 4096,
        max_issues: int = 5,
        max_issues_per_dimension: int = 2,
        max_sources: int = 20,
        context_chars: int = 8000,
        dimension_parse_retries: int = 1,
    ):
        """初始化 Red Agent。

        Args:
            policy: 任意实现了 __call__(messages:list) -> OpenAICompatibleDict 的对象。
            max_tokens: 每个维度评估的最大输出长度。
            max_issues: 单轮最多交给 Blue Agent 的问题数。
        """
        self.policy = policy
        self.max_tokens = max_tokens
        self.max_issues = max(1, int(max_issues))
        self.max_issues_per_dimension = max(1, int(max_issues_per_dimension))
        self.max_sources = max(1, int(max_sources))
        self.context_chars = max(1000, min(int(context_chars), 32000))
        self.dimension_parse_retries = max(0, min(int(dimension_parse_retries), 3))
        self.last_issue_stats = {"raw": 0, "deduplicated": 0, "selected": 0}
        self.last_retry_stats = {"attempted": 0, "recovered": 0, "exhausted": 0}

    @trace_agent(name="red_agent.attack", tags=["m5", "red", "adversarial"])
    async def attack(self, report: ResearchReport) -> RedVerdict:
        """对研究报告执行五维度攻击。

        执行流程：
        1. 并行（顺序 await 但可外部 gather）调用五个维度的评估 Prompt。
        2. 解析每个维度的 JSON 输出，提取分数和 issues。
        3. 汇总生成 RedVerdict。

        Args:
            report: 待审查的研究报告。

        Returns:
            RedVerdict: 包含五维度分数、overall_score 和 issues 列表。
        """
        dimension_scores: dict[Dimension, float] = {}
        all_issues: list[Issue] = []
        raw_feedbacks: list[str] = []
        errors: list[str] = []
        backend_unavailable = False
        retry_attempted = 0
        retry_recovered = 0
        retry_exhausted = 0

        # Bound every dimension prompt, but retain the report tail as well.  A
        # head-only excerpt systematically hid conclusions and references,
        # making Red's score an unreliable gate for long reports.
        content_truncated = self._excerpt(report.content, max_len=self.context_chars)
        
        sources_text = self._format_sources(report.sources, max_items=self.max_sources)

        for dim, prompt_template in DIMENSION_PROMPTS.items():
            # 使用安全替换，避免 report.content/sources_text 中的 { 被 format 误解析
            prompt = prompt_template
            prompt = prompt.replace("{query}", report.query)
            prompt = prompt.replace("{content}", content_truncated)
            prompt = prompt.replace("{sources}", sources_text)
            prompt += (
                "\n\n输出预算约束：issues 只保留本维度最高优先级的最多 "
                f"{self.max_issues_per_dimension} 个；不要枚举次要或重复问题。"
            )
            messages = [
                {"role": "system", "content": SYSTEM_RED_AGENT},
                {"role": "user", "content": prompt},
            ]

            dimension_ok = False
            last_error: Exception | None = None
            for attempt in range(self.dimension_parse_retries + 1):
                try:
                    # 临时调大 max_tokens 以容纳长输出
                    old_max = getattr(self.policy, "max_tokens", None)
                    if old_max is not None:
                        self.policy.max_tokens = self.max_tokens
                    resp = await self._call_policy(messages)
                    if getattr(resp, "get", None) and resp.get("status") == "failed":
                        backend_unavailable = True
                        raise RuntimeError(resp.get("error", "Red Agent backend unavailable"))

                    raw = (resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")) or ""
                    raw_feedbacks.append(
                        f"[{dim.value} attempt={attempt + 1}]\n{raw}\n"
                    )
                    try:
                        score, issues = self._parse_json_output(raw, dim)
                    except ValueError as exc:
                        last_error = exc
                        if attempt >= self.dimension_parse_retries:
                            retry_exhausted += 1
                            break
                        retry_attempted += 1
                        messages = [
                            {"role": "system", "content": SYSTEM_RED_AGENT},
                            {"role": "user", "content": prompt + (
                                "\n\n上一次输出无法解析。请只返回一个合法 JSON 对象，"
                                "不要使用 Markdown 代码块或附加解释。"
                            )},
                        ]
                        continue
                    dimension_scores[dim] = score
                    all_issues.extend(issues)
                    dimension_ok = True
                    if attempt:
                        retry_recovered += 1
                    break
                except Exception as exc:
                    # Backend/transport failures are not parse failures. Do
                    # not multiply requests against an unhealthy dependency.
                    last_error = exc
                    break
                finally:
                    if old_max is not None:
                        self.policy.max_tokens = old_max
            if not dimension_ok:
                # A failed dimension is a failed adversarial evaluation, not a
                # neutral/high score.  The loop will skip optimization.
                error = last_error or RuntimeError("unknown Red dimension error")
                errors.append(f"{dim.value}: {error}")
                dimension_scores[dim] = 0.0
                raw_feedbacks.append(f"[{dim.value}]\nERROR: {error}\n")
                if backend_unavailable:
                    break

        raw_issue_count = len(all_issues)
        deduplicated_issues = self._deduplicate_issues(all_issues)
        all_issues = sorted(
            deduplicated_issues,
            key=VerdictEngine.compute_priority,
            reverse=True,
        )[: self.max_issues]
        self.last_issue_stats = {
            "raw": raw_issue_count,
            "deduplicated": len(deduplicated_issues),
            "selected": len(all_issues),
        }
        self.last_retry_stats = {
            "attempted": retry_attempted,
            "recovered": retry_recovered,
            "exhausted": retry_exhausted,
        }

        status = "failed" if errors else "success"
        overall = VerdictEngine.compute_overall(dimension_scores) if not errors else 0.0
        return RedVerdict(
            dimension_scores=dimension_scores,
            overall_score=overall,
            issues=all_issues,
            raw_feedback="\n".join(raw_feedbacks),
            status=status,
            error="; ".join(errors),
            issue_stats=dict(self.last_issue_stats),
            retry_stats=dict(self.last_retry_stats),
        )

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

    def _select_issues(self, issues: list[Issue]) -> list[Issue]:
        """去重并只保留最高优先级问题，给 Blue 的调用量设置硬边界。"""
        unique = self._deduplicate_issues(issues)
        unique.sort(key=VerdictEngine.compute_priority, reverse=True)
        return unique[: self.max_issues]

    @staticmethod
    def _deduplicate_issues(issues: list[Issue]) -> list[Issue]:
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
        return unique

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """将来源列表格式化为文本，供 Prompt 使用。截断以避免上下文膨胀。"""
        if not sources:
            return "（无来源）"
        lines = []
        for i, s in enumerate(sources[:max_items], 1):
            title = s.get("title", "未知标题")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]  # 截断 snippet
            lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... 还有 {len(sources) - max_items} 个来源未显示")
        return "\n".join(lines)

    @staticmethod
    def _excerpt(content: str, max_len: int = 4000) -> str:
        """Return a head/tail excerpt for bounded adversarial prompts."""
        if len(content) <= max_len:
            return content
        marker = "\n\n[报告中间内容省略；请同时检查显示的开头和结尾]\n\n"
        available = max(0, max_len - len(marker))
        head_len = available // 2
        tail_len = available - head_len
        return content[:head_len] + marker + content[-tail_len:]

    def _parse_json_output(self, raw: str, dimension: Dimension) -> tuple[float, list[Issue]]:
        """解析模型 JSON 输出，提取分数和 issues。

        兼容策略：
        1. 先尝试从整个输出中提取第一个 JSON 对象。
        2. 如果失败，尝试用正则提取 ```json ... ``` 块。
        3. 如果仍失败，尝试修复常见 JSON 格式错误（如尾随逗号、单引号）。
        4. 如果仍失败，抛出解析错误，由 attack() 将本轮标记为 failed。

        Args:
            raw: 模型原始输出文本。
            dimension: 当前解析的维度，用于构造 Issue。

        Returns:
            (score, issues_list)
        """
        raw = raw.strip()
        if not raw:
            raise ValueError(f"empty {dimension.value} evaluation response")

        # 尝试 1: 直接解析整个文本
        try:
            data = json.loads(raw)
            return self._extract_from_dict(data, dimension)
        except json.JSONDecodeError:
            pass

        # 尝试 2: 提取 ```json 代码块
        code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        matches = code_block_pattern.findall(raw)
        for m in matches:
            try:
                data = json.loads(m.strip())
                return self._extract_from_dict(data, dimension)
            except json.JSONDecodeError:
                continue

        # 尝试 3: 提取第一个 { ... } 块（可能嵌套）
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return self._extract_from_dict(data, dimension)
            except json.JSONDecodeError:
                pass

        # 尝试 4: 修复常见 JSON 格式错误后重试
        fixed = self._fix_common_json_errors(raw)
        if fixed:
            for candidate in [fixed, fixed[fixed.find("{"):fixed.rfind("}")+1]]:
                try:
                    data = json.loads(candidate)
                    return self._extract_from_dict(data, dimension)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"invalid JSON response for {dimension.value}")

    def _fix_common_json_errors(self, raw: str) -> str | None:
        """修复常见的 JSON 格式错误。"""
        # 提取最外层的大括号内容
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = raw[start:end+1]
        
        # 修复 1: 移除注释
        text = re.sub(r"//.*?\n", "\n", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        
        # 修复 2: 单引号 → 双引号
        text = text.replace("'", '"')
        
        # 修复 3: 移除尾随逗号（对象/数组最后一个元素后的逗号）
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        
        # 修复 4: 修复未转义的换行符 inside 字符串（简单 heuristic）
        # 如果行内有 "..." 且中间有换行，可能导致解析失败
        
        return text

    def _extract_from_dict(self, data: dict, dimension: Dimension) -> tuple[float, list[Issue]]:
        """从解析后的 dict 中提取 score 和 issues。"""
        if not isinstance(data, dict) or "score" not in data:
            raise ValueError(f"{dimension.value} response missing score")
        score = float(data["score"])
        score = max(0.0, min(10.0, score))

        # severity/fix_type 容错映射
        sev_map = {
            "critical": Severity.CRITICAL, "严重": Severity.CRITICAL,
            "major": Severity.MAJOR, "重要": Severity.MAJOR, "较大": Severity.MAJOR,
            "minor": Severity.MINOR, "轻微": Severity.MINOR, "一般": Severity.MINOR,
        }
        fix_map = {
            "in_place": FixType.IN_PLACE, "就地修复": FixType.IN_PLACE, "修正": FixType.IN_PLACE,
            "search": FixType.SUPPLEMENTARY, "supplementary": FixType.SUPPLEMENTARY, "补充搜索": FixType.SUPPLEMENTARY, "搜索": FixType.SUPPLEMENTARY,
            "removal": FixType.REMOVAL, "删除": FixType.REMOVAL, "移除": FixType.REMOVAL,
        }

        issues: list[Issue] = []
        raw_issues = data.get("issues", [])
        # 兼容 issues 是字符串而不是列表的情况
        if isinstance(raw_issues, str):
            raw_issues = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            try:
                sev_raw = str(item.get("severity", "minor")).lower().strip()
                fix_raw = str(item.get("fix_type", "in_place")).lower().strip()
                sev = sev_map.get(sev_raw, Severity.MINOR)
                fix = fix_map.get(fix_raw, FixType.IN_PLACE)
                issues.append(
                    Issue(
                        severity=sev,
                        dimension=dimension,
                        description=str(item.get("description", "")),
                        location=str(item.get("location", "")),
                        fix_type=fix,
                        evidence=str(item.get("evidence", "")),
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
        return score, issues
