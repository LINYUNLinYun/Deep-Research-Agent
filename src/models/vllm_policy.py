"""
VLLM Policy — OpenAI API 封装

直接复用项目一实现，增加 from __future__ import annotations 以保持 Python 3.10+ 兼容性。
接口保持完全一致：
  - __call__(messages) -> OpenAICompatibleDict
  - set_tools(tools)
  - _truncate_messages(messages, max_chars)
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI
from ..utils.runtime_context import inject_runtime_context


__all__ = ["VLLMPolicy", "OpenAICompatibleDict"]


# 正则表达式：用于抠出 Qwen 在标签外输出废话时的工具指令
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# [质量过滤器] Assistant 绝不应该输出这些模板标记
# 如果检测到，整条 trajectory 标记为污染（复用 was_truncated 通道）
FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]


# 万能兼容类：让字典支持 .content 和 .tool_calls 访问
class OpenAICompatibleDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class VLLMPolicy:
    """VLLM Policy：封装 OpenAI 兼容 API（vLLM / OpenAI）。

    核心能力:
      - 消息格式清洗与合并（防止 vLLM 400）
      - 主动截断（保留 system + 最近交互，丢弃旧轮次）
      - 工具调用解析（原生 + 正则回退）
      - 错误分类处理（上下文超限抛异常，其他错误返回显式 failed 响应）
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
        thinking: bool | None = None,
        module_name: str | None = None,
    ):
        raw_client = OpenAI(base_url=base_url, api_key=api_key)
        # 如果 LangSmith 追踪开启，自动包装 client 以追踪所有 LLM 调用
        from ..utils.tracing import maybe_wrap_openai_client
        self.client = maybe_wrap_openai_client(raw_client)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tools = tools
        # None means "do not send provider-specific thinking flags".  This is
        # important for backends which reject the DeepSeek extension field.
        self.thinking = thinking
        self.module_name = module_name
        # [污染标记] 一旦发生过主动截断，整条 trajectory 作废
        self.was_truncated = False

    def set_tools(self, tools: list[dict]) -> None:
        """注册可用工具（OpenAI function calling schema）。"""
        self.tools = tools

    def _truncate_messages(self, messages: list, max_chars: int = 35000) -> list:
        """主动截断：保留 system、原始任务和最新完整工具交互。

        阈值 35000 字符 ≈ 11-12K content tokens（ratio 2.5-3.0 + overhead + tool metadata）。
        优先丢弃旧轮次；最新交互超限时缩减内容，保留工具调用配对。
        """
        system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        other_msgs = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]

        def _count_chars(msgs):
            total = 0
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                # 1. content 字符数
                total += len(str(m.get("content", "")))
                # 2. assistant message 的 tool_calls 中 arguments + name（这些是 token 大户但被遗漏）
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        func = tc.get("function", {})
                        total += len(str(func.get("arguments", "")))
                        total += len(str(func.get("name", "")))
                # 3. tool message 的 metadata（较短但也计入）
                if m.get("role") == "tool":
                    total += len(str(m.get("tool_call_id", "")))
                    total += len(str(m.get("name", "")))
            return total

        before_chars = _count_chars(messages)
        if before_chars <= max_chars:
            return messages

        self.was_truncated = True
        print(f"[TRUNCATE] Triggered: {before_chars} chars > {max_chars} threshold. n_msgs={len(messages)}")
        print(f"[TRUNCATE] System msgs: {len(system_msgs)}, Other msgs: {len(other_msgs)}")

        # Pin the original task and keep the latest complete interaction.
        task_index = next((i for i, m in enumerate(other_msgs)
                           if isinstance(m, dict) and m.get("role") == "user"), None)
        pinned = [other_msgs[task_index]] if task_index is not None else []
        remaining = [m for i, m in enumerate(other_msgs) if i != task_index]
        groups = []
        for message in remaining:
            if (isinstance(message, dict) and message.get("role") == "tool"
                    and groups and isinstance(groups[-1][0], dict)
                    and groups[-1][0].get("tool_calls")):
                groups[-1].append(message)
            else:
                groups.append([message])

        def assembled():
            return system_msgs + pinned + [m for group in groups for m in group]

        while len(groups) > 1 and _count_chars(assembled()) > max_chars:
            groups.pop(0)
        kept = [dict(m) if isinstance(m, dict) else m for m in assembled()]
        # Shrink payloads without deleting the latest tool group or mutating
        # caller-owned evidence. Protocol fields remain unchanged.
        candidates = [m for m in kept if isinstance(m, dict) and m.get("role") == "tool"]
        candidates += [m for m in kept if isinstance(m, dict) and m.get("role") not in ("system", "tool")]
        for message in candidates:
            excess = _count_chars(kept) - max_chars
            if excess <= 0:
                break
            content = str(message.get("content", ""))
            marker = "\n[CONTENT_TRUNCATED]"
            if len(content) > len(marker) + 1:
                new_len = max(1, len(content) - excess - len(marker))
                message["content"] = content[:new_len] + marker
        final_chars = _count_chars(kept)
        if final_chars > max_chars:
            raise RuntimeError("Context budget cannot fit system prompt and tool-call metadata")
        print(f"[TRUNCATE] Reduced to {final_chars} chars, preserved task and latest interaction")
        return kept

    def __call__(self, messages: list) -> OpenAICompatibleDict:
        """调用 LLM，返回 OpenAI 兼容格式消息。

        Args:
            messages: OpenAI 格式的消息列表。

        Returns:
            OpenAICompatibleDict: 包含 role, content, tool_calls 字段。
        """
        # 1. 深度清洗消息格式.  Runtime context is injected here so all
        # callers (including custom agents) receive the same date policy.
        messages = inject_runtime_context(messages)
        sanitized = []
        for m in messages:
            role, content = "user", ""
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                # 修复核心报错：处理 ['observation', '...'] 这种元组格式
                role = "user" if m[0] in ["observation", "user"] else "assistant"
                content = str(m[1])

            # 过滤环境内部泄露的 Task 对象信息，防止干扰模型
            if "task=Task(" in str(content):
                continue

            new_msg = {"role": role, "content": str(content)}
            # 保留 assistant 的 tool_calls 和 tool 的元数据，否则 vLLM 会报 400
            if role == "assistant" and isinstance(m, dict) and m.get("tool_calls"):
                new_msg["tool_calls"] = m["tool_calls"]
            # 保留 reasoning_content（DeepSeek 推理模型需要）
            if role == "assistant" and isinstance(m, dict) and m.get("reasoning_content"):
                new_msg["reasoning_content"] = m["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = m.get("tool_call_id", "") if isinstance(m, dict) else ""
                new_msg["name"] = m.get("name", "") if isinstance(m, dict) else ""

            # 合并连续的同角色消息，防止 vLLM 400 报错
            # 但包含 tool_calls / tool_call_id 的消息不能合并，否则字段会丢失
            can_merge = (
                sanitized
                and sanitized[-1]["role"] == role
                and role in ("user", "assistant")
                and "tool_calls" not in sanitized[-1]
                and "tool_calls" not in new_msg
                and "tool_call_id" not in new_msg
            )
            if can_merge:
                sanitized[-1]["content"] += "\n" + str(content)
            else:
                sanitized.append(new_msg)

        # 2. 主动截断（16K 约束下的质量过滤器）
        # 阈值 12-13K content tokens ≈ 40000 字符（ratio 2.8-3.2 + overhead）
        sanitized = self._truncate_messages(sanitized, max_chars=35000)

        # 3. 发送请求
        kwargs = dict(
            model=self.model_name,
            messages=sanitized,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        if self.thinking is not None:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}
            }
        try:
            # resp = self.client.chat.completions.create(**kwargs)
            # raw_msg = resp.choices[0].message
            # content = raw_msg.content or ""
            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            raw_msg = choice.message
            content = raw_msg.content or ""
            if getattr(choice, "finish_reason", None) == "length":
                self.was_truncated = True

            usage = getattr(resp, "usage", None)
            print(
                f"[LLM] {self.module_name or 'default'} → backend_model={self.model_name} "
                f"thinking={self.thinking} "
                f"finish_reason={getattr(choice, 'finish_reason', None)} "
                f"max_tokens={self.max_tokens} "
                f"prompt_tokens={getattr(usage, 'prompt_tokens', None)} "
                f"completion_tokens={getattr(usage, 'completion_tokens', None)}"
            )

            print(f"[LLM DEBUG] usage={usage}")
            print(f"[LLM DEBUG] content_len={len(content)}")
            print(
                f"[LLM DEBUG] reasoning_len="
                f"{len(getattr(raw_msg, 'reasoning_content', '') or '')}"
            )

            # 4. [FORBIDDEN] 检测 assistant 是否输出了不该出现的模板标记
            for forbidden in FORBIDDEN_TEMPLATE_TOKENS:
                if forbidden in content:
                    print(f"[FORBIDDEN] Detected '{forbidden}' in assistant content, marking trajectory as contaminated")
                    self.was_truncated = True
                    break

            # 5. 解析工具调用 (带正则回退)
            final_tool_calls = []
            raw_tool_calls = getattr(raw_msg, "tool_calls", None)
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    final_tool_calls.append(OpenAICompatibleDict(
                        id=tc.id, type="function",
                        function=OpenAICompatibleDict(name=tc.function.name, arguments=tc.function.arguments)
                    ))
            elif "<tool_call>" in content:
                matches = TOOL_CALL_PATTERN.findall(content)
                for i, m_str in enumerate(matches):
                    try:
                        d = json.loads(m_str.strip())
                        final_tool_calls.append(OpenAICompatibleDict(
                            id=f"manual_{i}", type="function",
                            function=OpenAICompatibleDict(name=d.get("name"), arguments=json.dumps(d.get("arguments", {})))
                        ))
                    except Exception:
                        continue

            # 6. 返回万能对象
            result = OpenAICompatibleDict(
                role="assistant",
                content=content,
                tool_calls=final_tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
                status="success",
            )
            if getattr(raw_msg, "reasoning_content", None):
                result["reasoning_content"] = raw_msg.reasoning_content
            return result

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            print(f"Policy Error: {err_str}")

            # Context 超限：确定性错误，立刻中止 trajectory（不继续浪费采样）
            if "maximum context length" in err_lower or "context length" in err_lower:
                n_msgs = len(messages)
                total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
                raise RuntimeError(
                    f"[CONTEXT_LENGTH_EXCEEDED] n_msgs={n_msgs}, est_chars={total_chars}: {err_str}"
                ) from e

            # Never turn a provider failure into a successful-looking assistant
            # response.  Returning an explicit failed result keeps existing
            # callers compatible while allowing agents to stop/replan safely.
            return OpenAICompatibleDict(
                role="assistant",
                content="",
                tool_calls=[],
                status="failed",
                error=err_str,
                finish_reason="error",
            )
