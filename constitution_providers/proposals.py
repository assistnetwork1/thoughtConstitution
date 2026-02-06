# constitution_providers/proposals.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence, Tuple

from constitution_engine.models.recommendation import RankedOption


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------
# Provider identity / metadata
# ----------------------------

@dataclass(frozen=True)
class ProviderMeta:
    """
    Identifies a provider implementation (human, LLM, retriever, simulator, etc.)
    Providers have NO authority. This is provenance only.
    """
    provider_id: str
    provider_version: str
    generated_at: datetime = field(default_factory=now_utc)


# ----------------------------
# Draft (non-authoritative) artifacts
# ----------------------------

@dataclass(frozen=True)
class DraftEvidence:
    """
    Evidence-only payload (retrievers).
    This is NOT an assertion of truth; it's an information/provenance anchor.
    """
    source_uri: str
    excerpt: Optional[str] = None
    retrieved_at: datetime = field(default_factory=now_utc)
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftInterpretation:
    """
    Interpretive hypothesis/meaning/story about observations/evidence.
    """
    text: str
    confidence: float  # 0..1 (non-Bayesian scalar container)
    uncertainty: float  # 0..1 (remaining fragility)
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftOption:
    """
    Proposed candidate action. Kernel will re-check all gating + constraints.
    """
    title: str
    description: str
    impact: float          # 0..1
    reversibility: float   # 0..1
    action_class: str      # "probe" | "limited" | "commit" (proposal only)
    confidence: float      # 0..1
    uncertainty: float     # 0..1
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftRanking:
    """
    Proposed ranking over draft options (by local IDs or titles).
    The kernel constructs the final Recommendation after validation + gating.
    """
    ordered_keys: Sequence[str]
    rationale: str
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftBundle:
    """
    The provider's *draft* output. Non-authoritative.
    This is the only thing providers are allowed to emit.
    """
    evidence: Tuple[DraftEvidence, ...] = ()
    interpretations: Tuple[DraftInterpretation, ...] = ()
    options: Tuple[DraftOption, ...] = ()
    ranking: Optional[DraftRanking] = None


@dataclass(frozen=True)
class DraftEnvelope:
    """
    Constitutional boundary object:
    provider -> (draft output + provenance) -> kernel-owned normalization.
    """
    meta: ProviderMeta
    draft: DraftBundle


# ------------------------------------------------------------
# Runner contract (hardened): no provider Recommendation objects
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProposalSet:
    """
    Runner-facing bundle of provider outputs.

    Constitutional rule:
    - Proposals are non-authoritative drafts.
    - Providers may NOT emit kernel Recommendation objects.
    - Providers may emit:
        • options
        • proposed_ranked_options (ranking input only)
        • optional rationale text
    - The kernel constructs the canonical Recommendation after gating/constraints.

    This class is the CURRENT contract used by:
    - constitution_providers.runner.run_provider(...)
    - providers implementing ProposalProvider.propose(...)
    """
    provider_id: str

    # Provider annotations (non-authoritative)
    notes: str = ""

    # Optional trace / debugging metadata (non-authoritative)
    meta: Mapping[str, object] = field(default_factory=dict)

    # Timestamp for audit correlation (doesn't require providers to supply it)
    generated_at: datetime = field(default_factory=now_utc)

    # Proposal content (kernel-model objects today)
    evidence: Tuple[object, ...] = ()
    observations: Tuple[object, ...] = ()
    interpretations: Tuple[object, ...] = ()
    options: Tuple[object, ...] = ()

    # Ranking input ONLY (no Recommendation objects)
    proposed_ranked_options: Tuple[RankedOption, ...] = ()
    proposed_rationale: Optional[str] = None
