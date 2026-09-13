"""Schemas exchanged by claim-level evidence verification.

The fields intentionally use plain serialisable values.  Reports and tool
trajectories are persisted as JSON in this project, so keeping these objects
free of model/tool classes makes them safe to log and pass between workers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(slots=True)
class SourceRecord:
    """Canonical, compression-safe source record used by synthesis."""

    source_id: str
    canonical_url: str
    title: str = ""
    source_span: str = ""
    source_date: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    source_cluster_id: str = ""
    task_id: str = ""
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceBundle:
    """Worker-independent evidence view retained across compression."""

    sources: list[SourceRecord] = field(default_factory=list)
    claims: list["Claim"] = field(default_factory=list)
    edges: list["ClaimEvidenceEdge"] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "claims": [claim.to_dict() for claim in self.claims],
            "edges": [edge.to_dict() for edge in self.edges],
            "open_questions": list(self.open_questions),
        }


@dataclass(slots=True)
class ClaimEvidenceEdge:
    """Auditable relation between one atomic claim and one source span."""

    claim_id: str
    source_id: str
    relation: str
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerificationStatus(str, Enum):
    """Verification outcome for a claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Evidence:
    """A source span that can support or contradict a claim.

    ``source_span`` should contain the smallest useful excerpt available from
    a browser/search tool.  It may be empty when only a URL is known; such an
    item is intentionally insufficient for a ``supported`` verdict.
    """

    claim_id: str = ""
    source_url: str = ""
    source_span: str = ""
    source_date: str = ""
    domain: str = ""
    verification_status: VerificationStatus = VerificationStatus.UNKNOWN
    confidence: float = 0.0
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.verification_status, VerificationStatus):
            try:
                self.verification_status = VerificationStatus(str(self.verification_status))
            except ValueError:
                self.verification_status = VerificationStatus.UNKNOWN
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verification_status"] = self.verification_status.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Evidence":
        values = dict(data or {})
        return cls(**{k: values[k] for k in cls.__dataclass_fields__ if k in values})


@dataclass(slots=True)
class Claim:
    """A report claim selected for verification."""

    claim_id: str
    text: str
    location: str = ""
    source_refs: list[str] = field(default_factory=list)
    risk: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Claim":
        values = dict(data)
        values.setdefault("claim_id", "")
        values.setdefault("text", "")
        values.setdefault("source_refs", [])
        return cls(**{k: values[k] for k in cls.__dataclass_fields__ if k in values})


@dataclass(slots=True)
class VerificationResult:
    """Evidence decision for one claim."""

    claim: Claim
    status: VerificationStatus = VerificationStatus.UNKNOWN
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            try:
                self.status = VerificationStatus(str(self.status))
            except ValueError:
                self.status = VerificationStatus.UNKNOWN
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))

    @property
    def claim_id(self) -> str:
        return self.claim.claim_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
        }
