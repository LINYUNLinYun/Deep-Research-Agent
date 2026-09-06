"""Small, dependency-free helpers for temporal evidence metadata."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def infer_source_date(item: dict[str, Any]) -> str | None:
    for key in ("source_date", "published_at", "published", "date", "created_at"):
        value = item.get(key)
        if value:
            match = re.search(r"(20\d{2})(?:[-/]?(\d{1,2})(?:[-/]?(\d{1,2}))?)?", str(value))
            if match:
                return "-".join(p for p in match.groups() if p)
    text = " ".join(str(item.get(k, "")) for k in ("title", "url", "snippet"))
    match = re.search(r"20\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", text)
    return match.group(0).replace("/", "-") if match else None


def annotate_search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add retrieval and source-date metadata without changing search content."""
    retrieved_at = datetime.now().astimezone().isoformat()
    payload = dict(payload)
    payload["retrieved_at"] = payload.get("retrieved_at", retrieved_at)
    query = str(payload.get("query", "")).lower()
    temporal_query = any(token in query for token in ("今年", "当前", "目前", "最新", "近期", "最近", "this year", "current", "latest", "recent"))
    current_year = datetime.now().astimezone().year
    items = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            items.append(item)
            continue
        enriched = dict(item)
        enriched.setdefault("retrieved_at", payload["retrieved_at"])
        enriched.setdefault("source_date", infer_source_date(enriched))
        source_date = enriched.get("source_date")
        if not temporal_query:
            relevance = "not_applicable"
        elif not source_date:
            relevance = "unknown"
        else:
            try:
                source_year = int(str(source_date)[:4])
                relevance = "current" if source_year == current_year else ("historical_context" if source_year < current_year else "future_dated")
            except (TypeError, ValueError):
                relevance = "unknown"
        enriched.setdefault("temporal_relevance", relevance)
        items.append(enriched)
    if "results" in payload:
        payload["results"] = items
    return payload


__all__ = ["infer_source_date", "annotate_search_payload"]
