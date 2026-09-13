"""Strict record/replay adapters for isolated Harness evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


class ReplayMismatchError(RuntimeError):
    pass


class RecordedToolError(RuntimeError):
    """An error intentionally captured in a complete replay fixture."""

    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def request_fingerprint(tool_name: str, args: Mapping[str, Any]) -> str:
    payload = f"{tool_name}\n{canonical_json(dict(args))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RecordReplayStore:
    """A frozen query-keyed fixture universe with strict consumption.

    Multiple entries for one fingerprint represent repeated responses.  Calls
    are ordered per fingerprint rather than globally so baseline and candidate
    policies may legitimately choose different search queries from the same
    frozen universe.
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self.entries = list(entries or [])
        self._by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._cursor: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            self._by_key[str(entry["fingerprint"])].append(entry)

    @classmethod
    def load(cls, path: str | Path) -> "RecordReplayStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("replay fixture must contain an entries list")
        return cls(entries)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json({"version": 1, "entries": self.entries}) + "\n", encoding="utf-8")

    def record(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        response: Any = None,
        error: str = "",
        latency: float = 0.0,
    ) -> None:
        fingerprint = request_fingerprint(tool_name, args)
        entry = {
            "tool": tool_name,
            "args": dict(args),
            "fingerprint": fingerprint,
            "sequence": len(self._by_key[fingerprint]),
            "response": response,
            "error": error,
            "latency": float(latency),
        }
        self.entries.append(entry)
        self._by_key[fingerprint].append(entry)

    def replay(self, tool_name: str, args: Mapping[str, Any]) -> Any:
        fingerprint = request_fingerprint(tool_name, args)
        values = self._by_key.get(fingerprint, [])
        index = self._cursor[fingerprint]
        if index >= len(values):
            raise ReplayMismatchError(
                f"unmatched replay request: tool={tool_name}, fingerprint={fingerprint}, args={canonical_json(args)}"
            )
        entry = values[index]
        self._cursor[fingerprint] += 1
        if entry.get("error"):
            # Recorded timeout/429/5xx responses belong to the frozen fixture
            # universe.  They are not missing fixture coverage and must remain
            # distinguishable from an unmatched request.
            raise RecordedToolError(f"recorded tool error: {entry['error']}")
        return copy.deepcopy(entry.get("response"))

    def reset(self) -> None:
        self._cursor.clear()

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json({"version": 1, "entries": self.entries}).encode("utf-8")).hexdigest()


class ReplayToolAdapter:
    """Tool proxy that never falls through to a network/backend call."""

    def __init__(self, tool: Any, store: RecordReplayStore) -> None:
        self._tool = tool
        self._store = store
        self.name = tool.name
        self.description = getattr(tool, "description", "")

    async def execute(self, **kwargs: Any) -> Any:
        return self._store.replay(self.name, kwargs)

    def get_openai_tool_schema(self) -> dict[str, Any]:
        return self._tool.get_openai_tool_schema()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)


class RecordToolAdapter:
    """Tool proxy that records exact responses and errors for later freezing."""

    def __init__(self, tool: Any, store: RecordReplayStore) -> None:
        self._tool = tool
        self._store = store
        self.name = tool.name
        self.description = getattr(tool, "description", "")

    async def execute(self, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self._tool.execute(**kwargs)
        except Exception as exc:
            self._store.record(
                self.name, kwargs, error=f"{type(exc).__name__}: {exc}", latency=time.perf_counter() - started
            )
            raise
        self._store.record(self.name, kwargs, response=response, latency=time.perf_counter() - started)
        return response

    def get_openai_tool_schema(self) -> dict[str, Any]:
        return self._tool.get_openai_tool_schema()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)


def attach_replay_tools(modules: dict[str, Any], store: RecordReplayStore) -> None:
    """Replace module tools before agents are acquired; no backend fallback."""
    original = list(modules.get("tools", []))
    wrapped = [ReplayToolAdapter(tool, store) for tool in original]
    replacements = {id(tool): adapter for tool, adapter in zip(original, wrapped)}
    modules["tools"] = wrapped
    pool = modules.get("agent_pool")
    if pool is not None:
        pool.tools_factory = lambda: list(wrapped)
        kwargs = getattr(pool, "agent_kwargs", {})
        tool_policy = kwargs.get("tool_policy")
        if tool_policy is not None:
            # Strict replay mismatches must not be hidden by retry/fallback.
            tool_policy.max_retries = 0

    # ``initialize_modules`` constructs the adversarial Blue Agent before the
    # replay store is attached.  It therefore owns direct references to the
    # original tools and can otherwise bypass the frozen fixture universe via
    # supplementary search.  Replace both its public list and cached search
    # handle with the same replay adapters used by AgentPool.
    adversarial = modules.get("adversarial")
    blue_agent = getattr(adversarial, "blue_agent", None)
    if blue_agent is not None:
        blue_tools = list(getattr(blue_agent, "tools", []) or [])
        blue_agent.tools = [replacements.get(id(tool), tool) for tool in blue_tools]
        cached_search = getattr(blue_agent, "_search_tool", None)
        if cached_search is not None:
            blue_agent._search_tool = replacements.get(id(cached_search), cached_search)
