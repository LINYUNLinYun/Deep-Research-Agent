"""Claim-level evidence verification primitives.

The verifier is deliberately independent from the orchestration layer.  It can
be used after synthesis, by an orchestrator replan policy, or in an evaluation
script without requiring an LLM.  A caller may provide a richer evidence
fetcher/policy later; the deterministic lexical verifier remains a useful,
auditable baseline and never claims that an unsupported claim is true.
"""

from .schemas import (
    Claim,
    Evidence,
    VerificationResult,
    VerificationStatus,
    SourceRecord,
    EvidenceBundle,
    ClaimEvidenceEdge,
)
from .verifier import EvidenceVerifier
from .ledger import EvidenceLedger

__all__ = [
    "Claim",
    "Evidence",
    "VerificationResult",
    "VerificationStatus",
    "EvidenceVerifier",
    "SourceRecord",
    "EvidenceBundle",
    "ClaimEvidenceEdge",
    "EvidenceLedger",
]
