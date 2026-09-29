"""Session-aware search control for Researcher agents.

The original search path called a backend directly for every tool call.  That
made parallel workers issue the same query repeatedly and caused the final
context to be dominated by duplicate URLs.  :class:`SearchController` keeps a
small, process-local session cache and performs deterministic query rewriting
when a query has low evidence novelty.

The controller deliberately does not call an LLM.  Rewriting is cheap,
deterministic, and safe to use in tests/offline mode; an application can still
provide richer ``search_hints`` in the context.
"""
from __future__ import annotations

import copy
import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = ["SearchController", "SearchControlStats"]


_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "at",
    "with", "about", "from", "this", "that", "最新", "如何", "什么", "以及",
    "的", "了", "和", "与", "及", "对", "中", "关于",
}


@dataclass
class SearchControlStats:
    """Observable counters used by telemetry and lightweight evaluations."""

    calls: int = 0
    backend_calls: int = 0
    cache_hits: int = 0
    rewritten_queries: int = 0
    duplicate_results: int = 0
    new_results: int = 0
    blocked_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "backend_calls": self.backend_calls,
            "cache_hits": self.cache_hits,
            "rewritten_queries": self.rewritten_queries,
            "duplicate_results": self.duplicate_results,
            "new_results": self.new_results,
            "blocked_calls": self.blocked_calls,
        }


class SearchController:
    """Control one logical research session's searches.

    Parameters are intentionally conservative so creating the controller with
    no configuration remains backwards compatible.  The instance is normally
    attached to the shared ``web_search`` tool by ``ResearcherAgent``; this is
    what gives independently scheduled workers a common deduplication scope.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        cache_ttl_seconds: float = 3600.0,
        max_cache_entries: int = 512,
        query_similarity_threshold: float = 0.82,
        novelty_threshold: float = 0.15,
        max_rewrites: int = 1,
        max_backend_calls: int = 64,
        policy: Any | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.cache_ttl_seconds = max(float(cache_ttl_seconds), 0.0)
        self.max_cache_entries = max(int(max_cache_entries), 16)
        self.policy = policy
        # A versioned policy owns evolvable thresholds while legacy
        # construction keeps the exact historical defaults/configuration.
        if policy is not None:
            query_similarity_threshold = policy.query_similarity_threshold
            novelty_threshold = policy.novelty_threshold
            max_rewrites = policy.max_rewrites
        self.query_similarity_threshold = min(max(float(query_similarity_threshold), 0.0), 1.0)
        self.novelty_threshold = min(max(float(novelty_threshold), 0.0), 1.0)
        self.max_rewrites = max(int(max_rewrites), 0)
        self.max_backend_calls = max(int(max_backend_calls), 1)

        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._query_history: dict[str, str] = {}
        self._seen_urls: set[str] = set()
        self._seen_content: set[str] = set()
        self._policy_decisions: list[dict[str, Any]] = []
        self._search_events: list[dict[str, Any]] = []
        # The controller is shared by all workers in a research run.  Policy
        # search-attempt limits therefore need a task-local counter; applying
        # them to ``stats.backend_calls`` would let the first worker exhaust
        # the allowance for every other task.
        self._task_backend_calls: dict[str, int] = {}
        self.stats = SearchControlStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool: Any,
        args: Mapping[str, Any] | None = None,
        *,
        execution_policy: Any | None = None,
        fallback_tools: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a web search with cache, rewrite, and cross-worker dedup.

        ``execution_policy`` is duck typed to avoid a hard dependency on the
        retry module.  It should expose ``execute(tool_name, tool, args, ...)``
        and returns the backend payload unchanged (plus telemetry metadata).
        """

        raw_args = dict(args or {})
        query = str(raw_args.get("query", "")).strip()
        if not query:
            return {"query": query, "results": [], "total": 0, "error": "query is required"}

        if not self.enabled:
            raw_args["query"] = query
            if execution_policy is not None:
                return await execution_policy.execute(
                    "web_search", tool, raw_args,
                    fallback_tools=fallback_tools, treat_empty_as_error=False,
                )
            return await tool.execute(**raw_args)

        self.stats.calls += 1
        # Versioned policies make their decision after observing the first
        # response.  The old pre-search heuristic is retained only when no
        # policy is configured.
        effective_query = query if self.policy is not None else self.rewrite_query(query, context=context)
        if effective_query != query:
            self.stats.rewritten_queries += 1
        raw_args["query"] = effective_query
        cache_key = self.query_fingerprint(effective_query)

        cached = self._get_cached(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            payload = copy.deepcopy(cached)
            payload["cache_hit"] = True
            payload["deduplicated"] = True
            payload["original_query"] = query
            payload["effective_query"] = effective_query
            # A cache hit contributes no new evidence, even though the cached
            # payload's first execution may have had novelty=1.0.
            payload["evidence_novelty"] = 0.0
            if self.policy is not None:
                decision = self.policy.decide(self._policy_signals(payload, context))
                self._attach_policy_decision(payload, decision)
            self._record_search_event(payload, query, effective_query, context=context)
            return payload

        if not self._reserve_backend_call(context):
            self.stats.blocked_calls += 1
            result = {
                "query": effective_query,
                "results": [],
                "total": 0,
                "original_query": query,
                "effective_query": effective_query,
                "hard_cap_reached": True,
                "stop_search_requested": True,
                "evidence_novelty": 0.0,
            }
            if self.policy is not None:
                decision = self.policy.decide(self._policy_signals(result, context))
                self._attach_policy_decision(result, decision)
            self._record_search_event(result, query, effective_query, context=context)
            return result
        if execution_policy is not None:
            result = await execution_policy.execute(
                "web_search",
                tool,
                raw_args,
                fallback_tools=fallback_tools,
                treat_empty_as_error=False,
            )
        else:
            try:
                result = await tool.execute(**raw_args)
            except Exception as exc:  # pragma: no cover - policy normally handles this
                result = {"query": effective_query, "results": [], "total": 0, "error": f"{type(exc).__name__}: {exc}"}

        if not isinstance(result, dict):
            result = {"query": effective_query, "results": [], "total": 0, "raw": result}
        result = copy.deepcopy(result)
        result.setdefault("query", effective_query)
        results = result.get("results")
        # Provider fallbacks such as ``arxiv_reader`` expose ``papers`` rather
        # than web-style ``results``. Normalize the common fields so a fallback
        # remains useful to the worker and participates in deduplication.
        if results is None and isinstance(result.get("papers"), list):
            results = [
                {
                    "title": str(p.get("title", "")) if isinstance(p, Mapping) else str(p),
                    "url": (
                        str(p.get("url") or p.get("pdf_url") or p.get("id", ""))
                        if isinstance(p, Mapping)
                        else ""
                    ),
                    "snippet": str(p.get("summary", p.get("abstract", ""))) if isinstance(p, Mapping) else str(p),
                }
                for p in result.get("papers", [])
            ]
            result["results"] = results
        if not isinstance(results, list):
            # A failed provider is cached only briefly by the caller's retry
            # policy; don't turn malformed responses into permanent cache hits.
            result.setdefault("results", [])
            result.setdefault("total", 0)
            result["original_query"] = query
            result["effective_query"] = effective_query
            self._record_search_event(result, query, effective_query, context=context)
            return result

        # Empty results are often a valid provider response rather than a
        # transport failure. Spend one bounded rewrite attempt before handing
        # the empty payload to the worker, even when this is the first query
        # in a session (there is no prior query to compare against yet).
        if self.policy is None and not results and self.max_rewrites > 0 and not result.get("error"):
            rewritten = self._rewrite_for_novelty(query, context)
            if rewritten and rewritten != effective_query and self._reserve_backend_call(context):
                self.stats.rewritten_queries += 1
                rewritten_args = dict(raw_args)
                rewritten_args["query"] = rewritten
                if execution_policy is not None:
                    rewritten_result = await execution_policy.execute(
                        "web_search",
                        tool,
                        rewritten_args,
                        fallback_tools=fallback_tools,
                        treat_empty_as_error=False,
                    )
                else:
                    try:
                        rewritten_result = await tool.execute(**rewritten_args)
                    except Exception as exc:  # pragma: no cover
                        rewritten_result = {"query": rewritten, "results": [], "error": f"{type(exc).__name__}: {exc}"}
                if isinstance(rewritten_result, dict) and isinstance(rewritten_result.get("results"), list):
                    if rewritten_result.get("results"):
                        result = copy.deepcopy(rewritten_result)
                        results = result.get("results", [])
                        effective_query = rewritten
                    result["rewritten_query"] = rewritten

        unique, duplicate_count = self._novel_results(results)
        # If every result was previously seen, retain the backend payload so
        # the model still has usable evidence, but mark it as non-novel.
        selected = unique if unique else results
        novelty = (len(unique) / max(len(results), 1)) if results else 0.0
        self.stats.duplicate_results += duplicate_count
        self.stats.new_results += len(unique)
        result["results"] = selected
        result["total"] = len(selected)
        result["original_query"] = query
        result["effective_query"] = effective_query
        if effective_query != query:
            result["rewritten_query"] = effective_query
        result["deduplicated"] = duplicate_count > 0
        result["duplicate_count"] = duplicate_count
        result["evidence_novelty"] = round(novelty, 4)

        # Policy execution is deliberately non-recursive: one initial search
        # yields one decision and at most one additional provider/tool call.
        if self.policy is not None:
            decision = self.policy.decide(self._policy_signals(result, context))
            self._attach_policy_decision(result, decision)
            if decision.action == "rewrite_uncovered_facets" and self.max_rewrites > 0:
                rewritten = self._rewrite_for_novelty(query, context)
                if rewritten and rewritten != effective_query and self._reserve_backend_call(context):
                    self.stats.rewritten_queries += 1
                    rewritten_args = dict(raw_args)
                    rewritten_args["query"] = rewritten
                    if execution_policy is not None:
                        second = await execution_policy.execute(
                            "web_search", tool, rewritten_args,
                            fallback_tools=fallback_tools, treat_empty_as_error=False,
                        )
                    else:
                        try:
                            second = await tool.execute(**rewritten_args)
                        except Exception as exc:  # pragma: no cover
                            second = {"query": rewritten, "results": [], "error": f"{type(exc).__name__}: {exc}"}
                    if isinstance(second, dict) and isinstance(second.get("results"), list):
                        second_results = second["results"]
                        second_unique, second_dupes = self._novel_results(second_results)
                        self.stats.duplicate_results += second_dupes
                        self.stats.new_results += len(second_unique)
                        if second_unique:
                            result["results"] = selected + second_unique
                            result["total"] = len(result["results"])
                        result["rewritten_query"] = rewritten
                        result["rewrite_evidence_novelty"] = round(
                            len(second_unique) / max(len(second_results), 1), 4
                        )
            elif decision.action == "switch_provider" and fallback_tools:
                fallback_name, fallback_tool, fallback_args = list(fallback_tools)[0]
                if not self._reserve_backend_call(context):
                    result["policy_fallback"] = {
                        "results": [],
                        "error_type": "budget",
                        "hard_cap_reached": True,
                        "stop_search_requested": True,
                    }
                elif execution_policy is not None:
                    second = await execution_policy.execute(
                        fallback_name, fallback_tool, fallback_args,
                        fallback_tools=None, treat_empty_as_error=False,
                    )
                else:
                    try:
                        second = await fallback_tool.execute(**fallback_args)
                    except Exception as exc:  # pragma: no cover
                        second = {"error": f"{type(exc).__name__}: {exc}"}
                result["policy_fallback"] = second
            elif decision.action == "invoke_numeric_verification":
                # EvidenceVerifier consumes this request downstream; the search
                # controller itself is intentionally unaware of Skill runtime.
                result["verification_requested"] = "verify_numeric_claim_skill"
            elif decision.action == "stop_search":
                result["stop_search_requested"] = True

        # A near-duplicate query with no new evidence gets one deterministic
        # rewrite.  The rewritten call is merged with the first response and
        # never recursively rewritten.
        if (
            self.policy is None
            and self.max_rewrites > 0
            and novelty <= self.novelty_threshold
            and results
            and effective_query == query
            and self._has_similar_query(effective_query)
        ):
            rewritten = self._rewrite_for_novelty(query, context)
            if rewritten and rewritten != effective_query and self._reserve_backend_call(context):
                self.stats.rewritten_queries += 1
                rewritten_args = dict(raw_args)
                rewritten_args["query"] = rewritten
                if execution_policy is not None:
                    second = await execution_policy.execute(
                        "web_search",
                        tool,
                        rewritten_args,
                        fallback_tools=fallback_tools,
                        treat_empty_as_error=False,
                    )
                else:
                    try:
                        second = await tool.execute(**rewritten_args)
                    except Exception as exc:  # pragma: no cover
                        second = {"query": rewritten, "results": [], "error": f"{type(exc).__name__}: {exc}"}
                if isinstance(second, dict):
                    second_results = second.get("results")
                    if isinstance(second_results, list):
                        second_unique, second_dupes = self._novel_results(second_results)
                        self.stats.duplicate_results += second_dupes
                        self.stats.new_results += len(second_unique)
                        if second_unique:
                            result["results"] = selected + second_unique
                            result["total"] = len(result["results"])
                            result["evidence_novelty"] = round(
                                (len(unique) + len(second_unique)) / max(len(results) + len(second_results), 1),
                                4,
                            )
                        result["rewritten_query"] = rewritten

        # Remember only successful-ish responses.  Errors are allowed through
        # to the retry policy on subsequent calls.
        if not result.get("error"):
            self._put_cache(cache_key, result)
            self._query_history[cache_key] = effective_query
        self._record_search_event(result, query, effective_query, context=context)
        return result

    def rewrite_query(self, query: str, *, context: Mapping[str, Any] | None = None) -> str:
        """Return a normalized query, optionally adding an uncovered facet."""

        normalized = " ".join(str(query).split()).strip()
        if not normalized:
            return normalized
        key = self.query_fingerprint(normalized)
        # Exact cache hits are intentionally not rewritten: the cache path is
        # cheaper and gives workers a consistent view of a search result.
        if key in self._cache:
            return normalized
        if not self._has_similar_query(normalized):
            return normalized
        return self._rewrite_for_novelty(normalized, context) or normalized

    def query_fingerprint(self, query: str) -> str:
        tokens = self._tokens(query)
        return " ".join(sorted(tokens))

    def snapshot(self) -> dict[str, Any]:
        """Return serializable search-control telemetry for a run."""

        return {
            "stats": self.stats.as_dict(),
            "cached_queries": len(self._cache),
            "seen_urls": len(self._seen_urls),
            "seen_content": len(self._seen_content),
            "policy": self.policy.metadata() if self.policy is not None else None,
            "policy_decisions": copy.deepcopy(self._policy_decisions),
            "task_backend_calls": copy.deepcopy(self._task_backend_calls),
            "events": copy.deepcopy(self._search_events),
        }

    def reset(self) -> None:
        """Start a fresh research run while retaining controller settings."""
        self._cache.clear()
        self._query_history.clear()
        self._seen_urls.clear()
        self._seen_content.clear()
        self._policy_decisions.clear()
        self._search_events.clear()
        self._task_backend_calls.clear()
        if self.policy is not None:
            self.policy.reset()
        self.stats = SearchControlStats()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _policy_signals(
        self,
        result: Mapping[str, Any],
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        context = context or {}
        results = result.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0
        duplicate_count = int(result.get("duplicate_count", 0) or 0)
        total_seen = result_count + duplicate_count
        unresolved = context.get("unresolved_claims", 0)
        if isinstance(unresolved, (list, tuple, set, dict)):
            unresolved = len(unresolved)
        try:
            unresolved = int(unresolved or 0)
        except (TypeError, ValueError):
            unresolved = 0
        remaining = context.get("remaining_search_budget", max(self.max_rewrites, 0))
        try:
            remaining = max(int(remaining), 0)
        except (TypeError, ValueError):
            remaining = max(self.max_rewrites, 0)
        return {
            "result_count": result_count,
            "duplicate_ratio": round(duplicate_count / max(total_seen, 1), 4),
            "evidence_novelty": float(result.get("evidence_novelty", 0.0) or 0.0),
            "unresolved_claims": unresolved,
            "search_attempts": self._task_backend_calls.get(self._task_key(context), 0),
            "remaining_search_budget": remaining,
            "provider_health": str(context.get("provider_health") or ("unhealthy" if result.get("error") else "healthy")),
            "query_type": str(context.get("query_type", "unknown")),
        }

    def _attach_policy_decision(self, result: dict[str, Any], decision: Any) -> None:
        record = decision.as_dict()
        result["_search_policy"] = copy.deepcopy(record)
        self._policy_decisions.append(record)

    def _record_search_event(
        self,
        result: Mapping[str, Any],
        original_query: str,
        effective_query: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist bounded, query-redacted evidence for offline policy mining."""

        def fingerprint(value: str) -> str:
            normalized = self.query_fingerprint(value)
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""

        results = result.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0
        policy = result.get("_search_policy", {})
        rewritten = str(result.get("rewritten_query", "") or "")
        event = {
            "sequence": len(self._search_events),
            "query_fingerprint": fingerprint(original_query),
            "effective_query_fingerprint": fingerprint(effective_query),
            "rewritten_query_fingerprint": fingerprint(rewritten),
            "query_changed": bool(effective_query != original_query),
            "cache_hit": bool(result.get("cache_hit")),
            "result_count": result_count,
            "duplicate_count": int(result.get("duplicate_count", 0) or 0),
            "evidence_novelty": float(result.get("evidence_novelty", 0.0) or 0.0),
            "rewrite_evidence_novelty": (
                float(result["rewrite_evidence_novelty"])
                if result.get("rewrite_evidence_novelty") is not None
                else None
            ),
            "rewrite_triggered": bool(rewritten),
            "hard_cap_reached": bool(result.get("hard_cap_reached")),
            "stop_search_requested": bool(result.get("stop_search_requested")),
            "provider_health": "unhealthy" if result.get("error") else "healthy",
            "policy_action": str(policy.get("action", "")) if isinstance(policy, Mapping) else "",
            "policy_rule_id": str(policy.get("rule_id", "")) if isinstance(policy, Mapping) else "",
            "stage": str((context or {}).get("stage", "worker")),
            "task_id": str((context or {}).get("task_id", "")),
            "facet_id": str((context or {}).get("facet_id", "")),
            "claim_ids": list((context or {}).get("claim_ids", []) or []),
            "source_cluster_ids": list((context or {}).get("source_cluster_ids", []) or []),
            "action": str((context or {}).get("action", "")),
            "estimated_value": (context or {}).get("estimated_value"),
            "task_search_attempts": self._task_backend_calls.get(self._task_key(context), 0),
            "run_backend_calls": self.stats.backend_calls,
        }
        self._search_events.append(event)

    @staticmethod
    def _task_key(context: Mapping[str, Any] | None) -> str:
        task_id = str((context or {}).get("task_id", "")).strip()
        return task_id or "__session__"

    def _reserve_backend_call(self, context: Mapping[str, Any] | None = None) -> bool:
        """Reserve one provider call before the first await in a worker.

        All workers share one controller.  Since this method contains no await,
        concurrent asyncio tasks cannot overbook either the run budget or a
        task's policy budget between the check and increment.
        """
        if self.stats.backend_calls >= self.max_backend_calls:
            return False
        task_key = self._task_key(context)
        task_calls = self._task_backend_calls.get(task_key, 0)
        if self.policy is not None and task_calls >= max(int(self.policy.max_search_attempts), 1):
            return False
        self.stats.backend_calls += 1
        self._task_backend_calls[task_key] = task_calls + 1
        return True

    def _tokens(self, text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized, flags=re.UNICODE)
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}

    def _has_similar_query(self, query: str) -> bool:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return False
        for previous in self._query_history.values():
            previous_tokens = self._tokens(previous)
            if not previous_tokens:
                continue
            overlap = len(query_tokens & previous_tokens) / max(len(query_tokens | previous_tokens), 1)
            if overlap >= self.query_similarity_threshold:
                return True
        return False

    def _rewrite_for_novelty(self, query: str, context: Mapping[str, Any] | None) -> str:
        context = context or {}
        candidates: list[str] = []
        hints = context.get("search_hints", [])
        if isinstance(hints, str):
            hints = [hints]
        if isinstance(hints, (list, tuple)):
            candidates.extend(str(h) for h in hints if h)
        for key in ("facet", "uncovered_facet", "description", "task_description"):
            value = context.get(key)
            if value:
                candidates.append(str(value))

        query_tokens = self._tokens(query)
        extra: list[str] = []
        for candidate in candidates:
            for token in re.findall(r"[\w\u4e00-\u9fff]+", unicodedata.normalize("NFKC", candidate).lower()):
                if token not in _STOPWORDS and len(token) > 1 and token not in query_tokens and token not in extra:
                    extra.append(token)
            if len(extra) >= 3:
                break
        if not extra:
            extra = ["independent sources"]
        return f"{query} {' '.join(extra[:3])}".strip()

    def _novel_results(self, results: list[Any]) -> tuple[list[Any], int]:
        unique: list[Any] = []
        duplicates = 0
        local_keys: set[str] = set()
        for item in results:
            if not isinstance(item, Mapping):
                key = self._content_key(str(item))
            else:
                url = self.canonical_url(str(item.get("url", "")))
                title = str(item.get("title", ""))
                snippet = str(item.get("snippet", item.get("summary", "")))
                key = url or self._content_key(f"{title} {snippet}")
            if not key or key in self._seen_urls or key in self._seen_content or key in local_keys:
                duplicates += 1
                continue
            local_keys.add(key)
            unique.append(item)
            if isinstance(item, Mapping) and item.get("url"):
                self._seen_urls.add(key)
            else:
                self._seen_content.add(key)
        return unique, duplicates

    @staticmethod
    def canonical_url(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlsplit(url.strip())
            if not parsed.netloc:
                return url.strip().lower()
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            path = parsed.path.rstrip("/") or "/"
            # Tracking parameters should not distinguish the same evidence.
            keep = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith(("utm_", "ref", "source"))]
            query = urlencode(sorted(keep))
            return urlunsplit((parsed.scheme.lower() or "https", netloc, path, query, ""))
        except Exception:
            return url.strip().lower()

    @staticmethod
    def _content_key(text: str) -> str:
        return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        item = self._cache.get(key)
        if item is None:
            return None
        timestamp, value = item
        if self.cache_ttl_seconds and time.monotonic() - timestamp > self.cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return value

    def _put_cache(self, key: str, value: dict[str, Any]) -> None:
        self._cache[key] = (time.monotonic(), copy.deepcopy(value))
        if len(self._cache) > self.max_cache_entries:
            oldest = min(self._cache.items(), key=lambda pair: pair[1][0])[0]
            self._cache.pop(oldest, None)
