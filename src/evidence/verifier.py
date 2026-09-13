"""Auditable claim-level evidence verifier.

This module intentionally has no network side effects.  Search/browser tools
already run in the worker layer; the verifier consumes their structured source
records and checks whether report claims have an attributable source span.  A
future caller can first fetch full pages and pass those spans as ``sources``.

The deterministic baseline is useful even without a judge model:

* high-risk claims (numbers, dates, comparisons and causal language) are
  selected first;
* source references such as ``[2]`` and URLs are honoured;
* token overlap and number consistency determine supported/contradicted/
  unknown, with conservative thresholds;
* no URL/snippet means never ``supported``.
"""
from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import replace
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from .schemas import Claim, Evidence, VerificationResult, VerificationStatus


SourceLike = Mapping[str, Any] | Evidence


class EvidenceVerifier:
    """Extract and verify claims against report sources.

    Args:
        policy: Optional callable receiving a claim and evidence list.  It may
            return a mapping (``status``, ``confidence``, ``reason``) or a
            :class:`VerificationResult`.  The lexical verifier is used when
            the policy is absent or fails, so a failed optional judge never
            turns unknown evidence into a false positive.
        max_claims: Upper bound per report to keep verification bounded.
        min_overlap: Minimum weighted token overlap for support.
    """

    _SENTENCE_RE = re.compile(r"(?<=[。！？!?\n])\s*|(?<=[.!?])\s+(?=[A-Z0-9])")
    _TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?")
    _NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
    _HIGH_RISK_RE = re.compile(
        r"(?:\d|%|percent|percentage|million|billion|year|date|排名|增长|下降|提高|降低|"
        r"超过|少于|同比|环比|因为|导致|因此|相比|最高|最低|first|last|increase|decrease|"
        r"because|therefore|versus)",
        re.IGNORECASE,
    )
    _NEGATION_RE = re.compile(
        r"(?:不|未|无|没有|否认|禁止|否定|(?<![A-Za-z])(?:not|no|never|without|false)(?![A-Za-z]))",
        re.IGNORECASE,
    )
    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "有", "是", "在", "和", "与", "的", "了", "为", "对", "中", "等", "一个",
    }

    def __init__(
        self,
        policy: Callable[..., Any] | None = None,
        *,
        max_claims: int = 50,
        min_overlap: float = 0.32,
    ) -> None:
        self.policy = policy
        self.max_claims = max(1, int(max_claims))
        self.min_overlap = max(0.05, min(1.0, float(min_overlap)))

    # ------------------------------------------------------------------ public
    def extract_claims(
        self,
        report: Any,
        *,
        max_claims: int | None = None,
    ) -> list[Claim]:
        """Extract deterministic, stable-id claims from report content.

        ``report`` may be a ``ResearchReport``, a mapping, or a plain string.
        Headlines and very short prose are skipped unless they carry a
        high-risk factual marker.  Stable IDs make verification traces
        comparable across replan rounds.
        """
        content = self._report_content(report)
        limit = max(1, int(max_claims or self.max_claims))
        claims: list[Claim] = []
        for idx, raw in enumerate(self._split_sentences(content), 1):
            text = raw.strip()
            if not text or len(text) < 8 or text.startswith("#"):
                continue
            risk = "high" if self._HIGH_RISK_RE.search(text) else "normal"
            # The verifier is most valuable on factual/high-risk claims; keep a
            # small sample of ordinary sentences to expose unsupported prose.
            if risk == "normal" and len(claims) >= limit // 2:
                continue
            refs = self._extract_refs(text)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            claims.append(
                Claim(
                    claim_id=f"claim_{digest}",
                    text=text,
                    location=f"sentence:{idx}",
                    source_refs=refs,
                    risk=risk,
                )
            )
            if len(claims) >= limit:
                break
        return claims

    def verify_sync(
        self,
        report: Any,
        sources: Sequence[SourceLike] | None = None,
        *,
        claims: Sequence[Claim] | None = None,
    ) -> list[VerificationResult]:
        """Synchronously verify claims using supplied source records."""
        source_items = self._normalise_sources(sources if sources is not None else self._report_sources(report))
        selected = list(claims) if claims is not None else self.extract_claims(report)
        return [self._verify_claim(claim, source_items) for claim in selected[: self.max_claims]]

    async def verify(
        self,
        report: Any,
        sources: Sequence[SourceLike] | None = None,
        *,
        claims: Sequence[Claim] | None = None,
    ) -> list[VerificationResult]:
        """Async entry point suitable for an orchestrator.

        If a policy is configured it can enrich the lexical result, but policy
        failures are recorded as reasons and the conservative baseline is
        returned.  This keeps verification useful in offline/test runs.
        """
        source_items = self._normalise_sources(
            sources if sources is not None else self._report_sources(report)
        )
        selected = list(claims) if claims is not None else self.extract_claims(report)
        selected = selected[: self.max_claims]
        results = [self._verify_claim(claim, source_items) for claim in selected]
        if self.policy is None:
            return results
        enriched: list[VerificationResult] = []
        for result in results:
            try:
                # The lexical verifier retains only its best source.  A
                # verification policy must instead see every attributable
                # source so it can detect conflicting numeric evidence.
                policy_sources = [
                    source
                    for source in source_items
                    if self._source_matches_refs(result.claim, source)
                ]
                candidate = self.policy(result.claim, policy_sources)
                if inspect.isawaitable(candidate):
                    candidate = await candidate
                enriched.append(self._merge_policy_result(result, candidate))
            except Exception as exc:  # pragma: no cover - defensive policy path
                result.reason = f"policy_failed: {exc}; {result.reason}".strip()
                enriched.append(result)
        return enriched

    async def verify_report(
        self,
        report: Any,
        sources: Sequence[SourceLike] | None = None,
        *,
        claims: Sequence[Claim] | None = None,
    ) -> list[VerificationResult]:
        """Named alias for callers that treat report verification as a stage."""
        return await self.verify(report, sources, claims=claims)

    def verify_report_sync(
        self,
        report: Any,
        sources: Sequence[SourceLike] | None = None,
        *,
        claims: Sequence[Claim] | None = None,
    ) -> list[VerificationResult]:
        """Synchronous counterpart of :meth:`verify_report`."""
        return self.verify_sync(report, sources, claims=claims)

    def verify_claim(
        self,
        claim: Claim,
        sources: Sequence[SourceLike] | None = None,
    ) -> VerificationResult:
        """Verify one already-extracted claim."""
        return self._verify_claim(claim, self._normalise_sources(sources or []))

    @staticmethod
    def summary(results: Iterable[VerificationResult]) -> dict[str, Any]:
        """Return compact metrics consumed by replan/evaluation code."""
        values = list(results)
        counts = {status.value: 0 for status in VerificationStatus}
        for item in values:
            counts[item.status.value] += 1
        total = len(values)
        return {
            "total_claims": total,
            "supported": counts[VerificationStatus.SUPPORTED.value],
            "contradicted": counts[VerificationStatus.CONTRADICTED.value],
            "unknown": counts[VerificationStatus.UNKNOWN.value],
            "support_rate": counts[VerificationStatus.SUPPORTED.value] / max(total, 1),
            "unsupported_rate": (
                (counts[VerificationStatus.CONTRADICTED.value] + counts[VerificationStatus.UNKNOWN.value])
                / max(total, 1)
            ),
        }

    # --------------------------------------------------------------- internals
    def _verify_claim(self, claim: Claim, sources: list[Evidence]) -> VerificationResult:
        if not sources:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNKNOWN,
                confidence=0.0,
                reason="no_source_records",
            )
        candidates = [source for source in sources if self._source_matches_refs(claim, source)]
        if not candidates:
            if claim.source_refs:
                # A citation that cannot be resolved must not silently fall
                # back to an unrelated source.  This is the key conservative
                # property that keeps citation coverage from becoming a false
                # support signal.
                return VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNKNOWN,
                    confidence=0.0,
                    reason="unresolved_source_reference",
                )
            candidates = sources
        scored: list[tuple[float, Evidence, bool]] = []
        claim_tokens = self._tokens(claim.text)
        # Citation markers such as ``[1]`` are references, not factual
        # numbers.  Counting them as claims creates deterministic false
        # contradictions whenever a source snippet omits the marker.
        claim_numbers = set(self._NUMBER_RE.findall(re.sub(r"\[\d+\]", "", claim.text)))
        claim_negated = bool(self._NEGATION_RE.search(claim.text))
        for source in candidates:
            text = " ".join(filter(None, [source.title, source.source_span]))
            source_tokens = self._tokens(text)
            overlap = self._weighted_overlap(claim_tokens, source_tokens)
            source_numbers = set(self._NUMBER_RE.findall(text))
            number_mismatch = bool(claim_numbers and source_numbers and not claim_numbers.issubset(source_numbers))
            source_negated = bool(self._NEGATION_RE.search(text))
            contradiction = number_mismatch or (claim_negated != source_negated and overlap >= self.min_overlap)
            scored.append((overlap, source, contradiction))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_overlap, best_source, contradiction = scored[0]
        if contradiction and best_overlap >= self.min_overlap:
            status = VerificationStatus.CONTRADICTED
            confidence = min(1.0, 0.55 + best_overlap * 0.45)
            reason = "source_span_conflicts_with_claim"
        elif best_overlap >= self.min_overlap and best_source.source_span.strip():
            status = VerificationStatus.SUPPORTED
            confidence = min(1.0, 0.45 + best_overlap * 0.55)
            reason = "source_span_supports_claim"
        else:
            status = VerificationStatus.UNKNOWN
            confidence = min(0.75, best_overlap)
            reason = "insufficient_entailment_or_source_span"
        best_source = replace(
            best_source,
            claim_id=claim.claim_id,
            verification_status=status,
            confidence=confidence,
        )
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=[best_source],
            reason=reason,
        )

    def _merge_policy_result(self, baseline: VerificationResult, candidate: Any) -> VerificationResult:
        if isinstance(candidate, VerificationResult):
            if candidate.claim.claim_id != baseline.claim.claim_id:
                candidate.claim = baseline.claim
            # A policy cannot upgrade a result without an attributable span.
            if candidate.status == VerificationStatus.SUPPORTED and not any(
                evidence.source_span.strip() for evidence in candidate.evidence
            ):
                return baseline
            return candidate
        if not isinstance(candidate, Mapping):
            return baseline
        status_raw = str(candidate.get("status", baseline.status.value)).lower()
        try:
            status = VerificationStatus(status_raw)
        except ValueError:
            status = baseline.status
        evidence = baseline.evidence
        if isinstance(candidate.get("evidence"), Sequence) and not isinstance(candidate.get("evidence"), (str, bytes)):
            evidence = self._normalise_sources(candidate["evidence"])
        if status == VerificationStatus.SUPPORTED and not any(item.source_span.strip() for item in evidence):
            status = baseline.status
        return VerificationResult(
            claim=baseline.claim,
            status=status,
            confidence=float(candidate.get("confidence", baseline.confidence) or baseline.confidence),
            evidence=evidence,
            reason=str(candidate.get("reason", baseline.reason)),
        )

    def _normalise_sources(self, sources: Sequence[SourceLike]) -> list[Evidence]:
        result: list[Evidence] = []
        for source_index, item in enumerate(sources, 1):
            if isinstance(item, Evidence):
                result.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("source_url", item.get("url", "")) or "")
            span = str(
                item.get("source_span", item.get("snippet", item.get("content", item.get("text", ""))))
                or ""
            )
            title = str(item.get("title", "") or "")
            domain = str(item.get("domain", "") or "")
            if not domain and url:
                domain = urlparse(url).netloc
            metadata = dict(item.get("metadata", {}) or {})
            metadata.setdefault("index", source_index)
            for key in ("source_id", "source_cluster_id", "citation_id", "content_hash", "task_id", "tool_name"):
                if item.get(key) is not None:
                    metadata.setdefault(key, item.get(key))
            result.append(
                Evidence(
                    claim_id=str(item.get("claim_id", "") or ""),
                    source_url=url,
                    source_span=span,
                    source_date=str(item.get("source_date", item.get("date", "")) or ""),
                    domain=domain,
                    title=title,
                    metadata=metadata,
                )
            )
        return result

    @staticmethod
    def _report_content(report: Any) -> str:
        if isinstance(report, str):
            return report
        if isinstance(report, Mapping):
            return str(report.get("content", "") or "")
        return str(getattr(report, "content", "") or "")

    @staticmethod
    def _report_sources(report: Any) -> Sequence[SourceLike]:
        if isinstance(report, Mapping):
            return report.get("sources", []) or []
        return getattr(report, "sources", []) or []

    def _split_sentences(self, content: str) -> list[str]:
        # Markdown line boundaries matter because a citation often follows the
        # sentence on the same line.  Strip bullets but retain citation markers.
        pieces: list[str] = []
        for paragraph in re.split(r"\n{2,}", content):
            pieces.extend(self._SENTENCE_RE.split(paragraph))
        cleaned = [re.sub(r"^\s*[-*]\s+", "", item).strip() for item in pieces if item.strip()]
        merged: list[str] = []
        for item in cleaned:
            # Chinese punctuation commonly leaves a following ``[N]`` as a
            # standalone split.  Keep that marker attached to the factual
            # sentence so source-reference filtering cannot be bypassed.
            if merged and re.fullmatch(r"(?:\[\d+\]\s*)+", item):
                merged[-1] = f"{merged[-1]}{item}"
            else:
                merged.append(item)
        return merged

    def _extract_refs(self, text: str) -> list[str]:
        refs = re.findall(r"\[(\d+)\]", text)
        refs.extend(re.findall(r"https?://[^\s)\]>]+", text))
        return list(dict.fromkeys(refs))

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in self._TOKEN_RE.findall(text)
            if token.lower() not in self._STOPWORDS
        }

    @staticmethod
    def _weighted_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(len(left), 1)

    @staticmethod
    def _source_matches_refs(claim: Claim, source: Evidence) -> bool:
        if not claim.source_refs:
            return True
        for ref in claim.source_refs:
            if ref.isdigit() and source.metadata.get("index") == int(ref):
                return True
            if ref.isdigit() and source.metadata.get("source_index") == int(ref):
                return True
            if ref.isdigit() and source.metadata.get("citation_id") == int(ref):
                return True
            if ref.startswith("http") and ref in source.source_url:
                return True
        return False
