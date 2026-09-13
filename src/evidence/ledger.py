"""Compression-safe evidence ledger built from worker tool trajectories."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from .schemas import EvidenceBundle, SourceRecord


class EvidenceLedger:
    """Normalize search, paper and browser outputs into canonical sources."""

    def __init__(self, *, max_sources_per_task: int = 12, max_span_chars: int = 4000) -> None:
        self.max_sources_per_task = max(1, int(max_sources_per_task))
        self.max_span_chars = max(240, int(max_span_chars))

    def collect(self, results: Sequence[Any]) -> EvidenceBundle:
        records: list[SourceRecord] = []
        seen: set[str] = set()
        for result in results:
            if str(getattr(getattr(result, "status", None), "value", "")) != "success":
                continue
            task_id = str(getattr(result, "task_id", ""))
            added = 0
            bundled = getattr(result, "evidence_bundle", {}) or {}
            bundled_sources = bundled.get("sources", []) if isinstance(bundled, Mapping) else []
            for source in bundled_sources if isinstance(bundled_sources, list) else []:
                if not isinstance(source, Mapping):
                    continue
                record = self._record(task_id, str(source.get("tool_name", "bundle")), source)
                if record is None or record.source_id in seen:
                    continue
                seen.add(record.source_id)
                records.append(record)
                added += 1
                if added >= self.max_sources_per_task:
                    break
            for step in getattr(result, "trajectory", []) or []:
                if not isinstance(step, Mapping) or step.get("role") != "tool":
                    continue
                tool_name = str(step.get("name", ""))
                result_payload = step.get("result")
                args = step.get("args") if isinstance(step.get("args"), Mapping) else {}
                for item in self._items(tool_name, result_payload, args):
                    record = self._record(task_id, tool_name, item)
                    if record is None or record.source_id in seen:
                        continue
                    seen.add(record.source_id)
                    records.append(record)
                    added += 1
                    if added >= self.max_sources_per_task:
                        break
                if added >= self.max_sources_per_task:
                    break
            # VERIFY workers return a typed claim verdict. Its exact spans are
            # first-class evidence for the next synthesis round rather than
            # being buried in free-form output.
            if added < self.max_sources_per_task:
                try:
                    verdict = json.loads(str(getattr(result, "output", "") or ""))
                except (TypeError, json.JSONDecodeError):
                    verdict = {}
                if isinstance(verdict, Mapping):
                    for item in verdict.get("evidence", []) if isinstance(verdict.get("evidence"), list) else []:
                        if not isinstance(item, Mapping):
                            continue
                        record = self._record(task_id, "verify_result", {
                            "url": item.get("url", ""),
                            "source_span": item.get("span", ""),
                            "title": item.get("title", ""),
                        })
                        if record is not None and record.source_id not in seen:
                            seen.add(record.source_id)
                            records.append(record)
                            added += 1
                            if added >= self.max_sources_per_task:
                                break
        return EvidenceBundle(sources=records)

    def catalog(self, results: Sequence[Any]) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for index, record in enumerate(self.collect(results).sources, 1):
            catalog.append({
                "citation_id": index,
                "source_id": record.source_id,
                "url": record.canonical_url,
                "canonical_url": record.canonical_url,
                "title": record.title,
                "snippet": record.source_span[:500],
                "source_span": record.source_span,
                "source_date": record.source_date,
                "retrieved_at": record.retrieved_at,
                "content_hash": record.content_hash,
                "source_cluster_id": record.source_cluster_id,
                "task_id": record.task_id,
                "tool_name": record.tool_name,
                "metadata": dict(record.metadata),
            })
        return catalog

    def _items(self, tool_name: str, payload: Any, args: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            for field in ("results", "papers", "sources", "evidence"):
                items = payload.get(field)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, Mapping)]
            if payload.get("url") or payload.get("canonical_url"):
                return [payload]
            return []
        if tool_name == "browser" and isinstance(payload, str) and payload.strip() and not payload.startswith("[Browser"):
            return [{"url": str(args.get("url", "")), "content": payload}]
        return []

    def _record(self, task_id: str, tool_name: str, item: Mapping[str, Any]) -> SourceRecord | None:
        url = str(item.get("canonical_url") or item.get("url") or item.get("pdf_url") or "").strip()
        span = str(
            item.get("source_span") or item.get("content") or item.get("text")
            or item.get("snippet") or item.get("summary") or item.get("abstract") or ""
        ).strip()[: self.max_span_chars]
        title = str(item.get("title", "") or "").strip()
        if not url or not (span or title):
            return None
        canonical = self._canonical_url(url)
        content_hash = hashlib.sha256(span.encode("utf-8")).hexdigest() if span else ""
        source_id = "src_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        host = urlsplit(canonical).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return SourceRecord(
            source_id=source_id,
            canonical_url=canonical,
            title=title,
            source_span=span,
            source_date=str(item.get("source_date") or item.get("published") or item.get("date") or ""),
            retrieved_at=str(item.get("retrieved_at") or datetime.now().astimezone().isoformat()),
            content_hash=content_hash,
            source_cluster_id="domain:" + host,
            task_id=task_id,
            tool_name=tool_name,
            metadata={"raw_url": url},
        )

    @staticmethod
    def _canonical_url(url: str) -> str:
        from src.tools.search_controller import SearchController

        return SearchController.canonical_url(url)
