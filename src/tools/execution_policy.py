"""Reliable tool execution policy.

Tool calls are external RPCs in practice: timeouts, rate limits, malformed
arguments, and temporary provider failures are all normal.  This module keeps
retry/fallback/circuit-breaker behavior out of individual tools so every
Researcher agent follows the same policy.
"""
from __future__ import annotations

import asyncio
import inspect
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["ToolExecutionPolicy", "ToolAttempt", "CircuitState"]


@dataclass
class ToolAttempt:
    tool: str
    attempt: int
    ok: bool
    error_type: str = ""
    error: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "attempt": self.attempt,
            "ok": self.ok,
            "error_type": self.error_type,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None
    successes: int = 0


class ToolExecutionPolicy:
    """Execute tools with bounded retry, fallback, and circuit breaking.

    ``max_retries`` is the number of retries *after* the initial attempt.  A
    retry is made only for transient errors (timeouts, 429/5xx, connection
    failures, and empty-result responses).  Invalid arguments and auth errors
    are permanent and immediately move to a fallback provider when available.
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        max_retry_delay: float = 8.0,
        circuit_failure_threshold: int = 3,
        circuit_cooldown: float = 30.0,
        timeout_seconds: float | None = None,
        jitter: float = 0.0,
    ) -> None:
        self.max_retries = max(int(max_retries), 0)
        self.retry_delay = max(float(retry_delay), 0.0)
        self.max_retry_delay = max(float(max_retry_delay), self.retry_delay)
        self.circuit_failure_threshold = max(int(circuit_failure_threshold), 1)
        self.circuit_cooldown = max(float(circuit_cooldown), 0.0)
        self.timeout_seconds = timeout_seconds if timeout_seconds is None else max(float(timeout_seconds), 0.0)
        self.jitter = max(float(jitter), 0.0)
        self._circuits: dict[str, CircuitState] = {}
        self._telemetry: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        tool: Any,
        args: Mapping[str, Any] | None = None,
        *,
        fallback_tools: Any | None = None,
        treat_empty_as_error: bool = True,
    ) -> Any:
        """Call ``tool.execute`` and return its native payload.

        ``fallback_tools`` accepts either a mapping ``name -> tool`` or a
        sequence of ``(name, tool)`` / ``(name, tool, args)`` tuples.  A
        fallback can therefore adapt arguments for providers with a different
        schema without changing this policy.
        """

        primary_args = dict(args or {})
        candidates = [(tool_name, tool, primary_args)]
        candidates.extend(self._normalize_fallbacks(fallback_tools, primary_args))
        attempts: list[ToolAttempt] = []
        last_error: tuple[str, str] = ("unknown", "tool execution failed")

        for candidate_name, candidate_tool, candidate_args in candidates:
            if candidate_tool is None:
                continue
            if self._circuit_open(candidate_name):
                attempts.append(ToolAttempt(candidate_name, 0, False, "circuit_open", "circuit breaker open"))
                last_error = ("circuit_open", f"Tool '{candidate_name}' circuit is open")
                continue

            for attempt_no in range(self.max_retries + 1):
                started = time.perf_counter()
                try:
                    value = candidate_tool.execute(**candidate_args)
                    if inspect.isawaitable(value):
                        if self.timeout_seconds is not None:
                            value = await asyncio.wait_for(value, timeout=self.timeout_seconds)
                        else:
                            value = await value
                    latency = (time.perf_counter() - started) * 1000.0
                    error_type, error = self._result_error(value, treat_empty_as_error=treat_empty_as_error)
                    if error_type:
                        attempts.append(ToolAttempt(candidate_name, attempt_no, False, error_type, error, latency))
                        self._record_failure(candidate_name)
                        last_error = (error_type, error)
                        if not self._retryable(error_type) or attempt_no >= self.max_retries:
                            break
                        await self._sleep_before_retry(attempt_no)
                        continue

                    attempts.append(ToolAttempt(candidate_name, attempt_no, True, latency_ms=latency))
                    self._record_success(candidate_name)
                    self._record_telemetry(tool_name, attempts)
                    # Attach metadata only to mutable mapping responses; raw
                    # strings/lists remain backwards compatible.
                    if isinstance(value, dict):
                        value = dict(value)
                        value.setdefault("_tool_execution", {
                            "requested_tool": tool_name,
                            "selected_tool": candidate_name,
                            "attempts": [a.as_dict() for a in attempts],
                        })
                    return value
                except Exception as exc:
                    latency = (time.perf_counter() - started) * 1000.0
                    error_type = self.classify_error(exc)
                    error = f"{type(exc).__name__}: {exc}"
                    attempts.append(ToolAttempt(candidate_name, attempt_no, False, error_type, error, latency))
                    self._record_failure(candidate_name)
                    last_error = (error_type, error)
                    if not self._retryable(error_type) or attempt_no >= self.max_retries:
                        break
                    await self._sleep_before_retry(attempt_no)

            # Permanent or exhausted primary failures fall through to the next
            # candidate.  Fallbacks never inherit the primary's retry count.

        self._record_telemetry(tool_name, attempts)
        return {
            "error": last_error[1],
            "error_type": last_error[0],
            "_tool_execution": {
                "requested_tool": tool_name,
                "selected_tool": None,
                "attempts": [a.as_dict() for a in attempts],
            },
        }

    def classify_error(self, error: Any) -> str:
        """Classify exceptions/provider payloads into stable error classes."""

        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        text = str(error).lower()
        if any(token in text for token in ("invalid argument", "missing required", "validation", "bad request", "400")):
            return "invalid_args"
        if any(token in text for token in ("api key", "authentication", "unauthorized", "forbidden", "401", "403")):
            return "auth"
        if any(token in text for token in ("429", "rate limit", "too many requests", "quota")):
            return "rate_limit"
        if any(token in text for token in ("500", "502", "503", "504", "server error", "service unavailable")):
            return "server"
        if any(token in text for token in ("connection", "connect", "network", "dns", "temporarily", "reset by peer")):
            return "network"
        if any(token in text for token in ("empty result", "no results", "no usable results")):
            return "empty"
        if isinstance(error, Mapping) and error.get("error"):
            return self.classify_error(error.get("error"))
        return "unknown"

    def is_retryable(self, error: Any) -> bool:
        return self._retryable(self.classify_error(error))

    def snapshot(self) -> dict[str, Any]:
        return {
            "circuits": {
                name: {
                    "failures": state.failures,
                    "opened": state.opened_at is not None,
                    "successes": state.successes,
                }
                for name, state in self._circuits.items()
            },
            "recent": self._telemetry[-50:],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _result_error(self, value: Any, *, treat_empty_as_error: bool = True) -> tuple[str, str]:
        if isinstance(value, Mapping) and value.get("error"):
            error = str(value.get("error"))
            return self.classify_error(error), error
        if value is None:
            return "empty", "empty result"
        if isinstance(value, Mapping):
            if treat_empty_as_error and "results" in value and not value.get("results") and not value.get("papers") and not value.get("content"):
                # Empty search results are transient by default.  A caller can
                # return an explicit ``allow_empty`` flag to treat them as a
                # successful answer.
                if not value.get("allow_empty"):
                    return "empty", "empty result"
            if treat_empty_as_error and "papers" in value and not value.get("papers") and not value.get("content"):
                if not value.get("allow_empty"):
                    return "empty", "empty result"
        return "", ""

    def _retryable(self, error_type: str) -> bool:
        return error_type in {"timeout", "rate_limit", "server", "network", "empty", "unknown"}

    async def _sleep_before_retry(self, attempt_no: int) -> None:
        delay = min(self.retry_delay * (2**attempt_no), self.max_retry_delay)
        if self.jitter:
            delay += random.uniform(0.0, self.jitter)
        if delay > 0:
            await asyncio.sleep(delay)

    def _record_failure(self, name: str) -> None:
        state = self._circuits.setdefault(name, CircuitState())
        state.failures += 1
        if state.failures >= self.circuit_failure_threshold:
            state.opened_at = time.monotonic()

    def _record_success(self, name: str) -> None:
        state = self._circuits.setdefault(name, CircuitState())
        state.failures = 0
        state.opened_at = None
        state.successes += 1

    def _circuit_open(self, name: str) -> bool:
        state = self._circuits.get(name)
        if state is None or state.opened_at is None:
            return False
        if self.circuit_cooldown <= 0 or time.monotonic() - state.opened_at >= self.circuit_cooldown:
            state.opened_at = None
            state.failures = 0
            return False
        return True

    def _normalize_fallbacks(self, fallbacks: Any, primary_args: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
        if not fallbacks:
            return []
        normalized: list[tuple[str, Any, dict[str, Any]]] = []
        if isinstance(fallbacks, Mapping):
            iterable = [(name, tool) for name, tool in fallbacks.items()]
        else:
            iterable = list(fallbacks)
        for item in iterable:
            if isinstance(item, tuple) or isinstance(item, list):
                if len(item) == 2:
                    name, tool = item
                    args = dict(primary_args)
                elif len(item) >= 3:
                    name, tool, args = item[:3]
                    args = dict(args or {})
                else:
                    continue
            else:
                continue
            normalized.append((str(name), tool, args))
        return normalized

    def _record_telemetry(self, requested: str, attempts: Sequence[ToolAttempt]) -> None:
        self._telemetry.append({
            "requested_tool": requested,
            "attempts": [a.as_dict() for a in attempts],
            "timestamp": time.time(),
        })
        if len(self._telemetry) > 200:
            del self._telemetry[:-200]
