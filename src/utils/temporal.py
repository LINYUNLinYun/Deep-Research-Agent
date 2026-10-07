"""Small, dependency-free helpers for temporal evidence metadata."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any


def infer_source_date(item: dict[str, Any]) -> str | None:
    def full_date(text: str) -> str | None:
        match = re.search(r"(?<!\d)(20\d{2})[-/年]?(\d{2}|\d(?=[-/月]))[-/月]?(\d{2}|\d(?!\d))(?:日)?(?!\d)", text)
        if match:
            try:
                return date(*(int(value) for value in match.groups())).isoformat()
            except ValueError:
                return None
        return None

    for key in ("source_date", "published_at", "published", "date", "created_at"):
        value = item.get(key)
        if value:
            exact = full_date(str(value))
            if exact:
                return exact
            match = re.search(r"(20\d{2})(?:[-/]?(\d{1,2})(?:[-/]?(\d{1,2}))?)?", str(value))
            if match:
                # A bare year inferred by an earlier tool annotation must not
                # hide a precise publication date available in the URL/title.
                fallback = "-".join(p for p in match.groups() if p)
                if len(fallback) > 4:
                    return fallback
                break
    else:
        fallback = None
    for key in ("url", "title", "snippet", "source_span"):
        exact = full_date(str(item.get(key, "")))
        if exact:
            return exact
    text = " ".join(str(item.get(k, "")) for k in ("title", "url", "snippet"))
    match = re.search(r"20\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", text)
    return fallback or (match.group(0).replace("/", "-") if match else None)


def temporal_relevance(query: str, source_date: str | None, *, today: date | None = None) -> str:
    """Keep relative-month queries distinct from a whole-calendar-year window."""
    query = query.lower()
    if not any(token in query for token in ("今年", "当前", "目前", "最新", "近期", "最近", "近一个月", "this year", "current", "latest", "recent", "past month", "last month")):
        return "not_applicable"
    if not source_date:
        return "unknown"
    today = today or datetime.now().astimezone().date()
    try:
        year = int(source_date[:4])
        month_query = any(token in query for token in ("近一个月", "最近一个月", "past month", "last month"))
        if month_query:
            if len(source_date) < 10:
                return "historical_context" if year < today.year else "unknown"
            published = date.fromisoformat(source_date[:10])
            if published > today:
                return "future_dated"
            return "current" if published >= today - timedelta(days=30) else "historical_context"
        return "current" if year == today.year else ("historical_context" if year < today.year else "future_dated")
    except (TypeError, ValueError):
        return "unknown"


def annotate_search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add retrieval and source-date metadata without changing search content."""
    retrieved_at = datetime.now().astimezone().isoformat()
    payload = dict(payload)
    payload["retrieved_at"] = payload.get("retrieved_at", retrieved_at)
    query = str(payload.get("query", "")).lower()
    items = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            items.append(item)
            continue
        enriched = dict(item)
        enriched.setdefault("retrieved_at", payload["retrieved_at"])
        enriched["source_date"] = infer_source_date(enriched)
        source_date = enriched.get("source_date")
        enriched["temporal_relevance"] = temporal_relevance(query, source_date)
        items.append(enriched)
    if "results" in payload:
        payload["results"] = items
    return payload


__all__ = ["infer_source_date", "annotate_search_payload", "temporal_relevance"]
