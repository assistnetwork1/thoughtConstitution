# constitution_providers/proposals.py
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence, Tuple


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------
# Provider identity / metadata
# ----------------------------

@dataclass(frozen=True)
class ProviderMeta:
    """
    Identifies a provider implementation (human, LLM, retriever, simulator, etc.)

    Providers have NO authority.
    This is provenance and audit metadata only.
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

    Constitutional meaning:
    - This is NOT an assertion of truth.
    - It's an information/provenance anchor (what was retrieved, from where).
    """
    source_uri: str
    excerpt: Optional[str] = None
    retrieved_at: datetime = field(default_factory=now_utc)
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftInterpretation:
    """
    Interpretive hypothesis/meaning/story about observations/evidence.

    Note:
    - confidence and uncertainty are non-Bayesian scalar containers in [0,1].
    """
    text: str
    confidence: float  # 0..1 (support strength container)
    uncertainty: float  # 0..1 (remaining fragility)
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftOption:
    """
    Proposed candidate action.

    Constitutional meaning:
    - Providers propose; kernel governs.
    - Kernel will re-check gating + constraints.
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
    Proposed ranking over draft options (by local keys).

    Constitutional meaning:
    - This is ranking input only.
    - The kernel constructs the canonical Recommendation after gating/constraints.

    ordered_keys:
    - Keys are provider-local identifiers. Common patterns:
      * DraftOption.title values (if unique), OR
      * explicit provider IDs (if provider includes them in meta)
    """
    ordered_keys: Sequence[str]
    rationale: str
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftBundle:
    """
    The provider's *draft* output. Non-authoritative.

    Constitutional rule:
    - This is the only thing providers are allowed to emit at the draft layer.
    - Kernel-owned artifacts (Recommendation, ChoiceRecord, Outcome, Review, Calibration, Override)
      are prohibited.
    """
    evidence: Tuple[DraftEvidence, ...] = ()
    interpretations: Tuple[DraftInterpretation, ...] = ()
    options: Tuple[DraftOption, ...] = ()
    ranking: Optional[DraftRanking] = None


@dataclass(frozen=True)
class DraftEnvelope:
    """
    Constitutional boundary object:
      provider -> (draft output + provenance) -> kernel normalization.

    This object is useful for:
    - audit
    - provenance threading
    - deterministic replay
    """
    meta: ProviderMeta
    draft: DraftBundle


# ------------------------------------------------------------
# Runner contract (hardened): ProposalSet (provider_rules compatible)
# WITH backwards-compatible legacy fields for existing stubs/tests
# ------------------------------------------------------------

@dataclass(frozen=True)
class ProposalUncertainty:
    """
    Provider-side uncertainty container.
    Non-Bayesian scalar: level ∈ [0,1].
    """
    level: float


@dataclass(frozen=True)
class ProposedInterpretation:
    interpretation_id: str
    info_type: str  # e.g., "hypothesis", "explanation", "frame"
    text: str

    confidence: float
    uncertainty: ProposalUncertainty
    evidence_refs: Tuple[str, ...]
    limits: str

    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedOption:
    option_id: str
    kind: str
    title: str
    description: str
    action_class: str

    impact: float
    reversibility: float

    confidence: float
    uncertainty: ProposalUncertainty
    evidence_refs: Tuple[str, ...]
    limits: str

    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedRankedOption:
    """
    Provider proposal: ranking input (NOT a kernel Recommendation).

    option_ref references ProposedOption.option_id
    """
    rank: int
    option_ref: str
    rationale: str

    # OPTIONAL but recommended:
    # include title to support stable mapping when canonicalization dedupes options
    title: Optional[str] = None

    confidence: float = 0.5
    uncertainty: ProposalUncertainty = field(default_factory=lambda: ProposalUncertainty(level=0.5))
    evidence_refs: Tuple[str, ...] = ()
    limits: str = ""

    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OverrideSuggestion:
    invariant_id: str
    reason: str
    scope: str  # e.g., "episode_only", "recommendation_only"

    confidence: float
    uncertainty: ProposalUncertainty
    evidence_refs: Tuple[str, ...]
    limits: str

    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalSampling:
    """
    Provider sampling metadata.
    Required by provider_rules: temperature must be present and numeric.
    """
    temperature: float = 0.0
    meta: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalSet:
    """
    Runner-facing bundle of provider outputs (provider_rules compatible).

    Backwards compatibility:
    - Legacy stubs/tests still pass:
        notes, evidence, observations, interpretations, options,
        proposed_ranked_options, proposed_rationale
    - We accept them and normalize into the provider_rules fields:
        evidence_threads, ranked_options
    """
    # ---- Required header fields (provider_rules) ----
    provider_id: str
    model_id: str = "stub_model"
    run_id: str = "stub_run"

    sampling: ProposalSampling = field(default_factory=ProposalSampling)
    limits: str = "stub_limits"

    # ---- Optional trace/debug metadata (non-authoritative) ----
    meta: Mapping[str, object] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=now_utc)

    # ---- Provider_rules fields ----
    evidence_threads: Tuple[object, ...] = ()
    interpretations: Tuple[object, ...] = ()
    options: Tuple[object, ...] = ()
    ranked_options: Tuple[object, ...] = ()
    override_suggestions: Tuple[object, ...] = ()

    # ---- Legacy compatibility fields (accepted; normalized) ----
    notes: str = ""
    evidence: Tuple[object, ...] = ()
    observations: Tuple[object, ...] = ()
    proposed_ranked_options: Tuple[object, ...] = ()
    proposed_rationale: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize legacy evidence -> evidence_threads
        if not self.evidence_threads and self.evidence:
            object.__setattr__(self, "evidence_threads", self.evidence)

        # Normalize legacy proposed_ranked_options -> ranked_options
        if not self.ranked_options and self.proposed_ranked_options:
            object.__setattr__(self, "ranked_options", self.proposed_ranked_options)


# Small helper (optional) for defensive construction in tests/providers
def make_dataclass(cls: type[Any], **kwargs: Any) -> Any:
    """
    Create dataclass instances defensively by filtering unknown kwargs.
    Useful in tests when models evolve.
    """
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)
    return cls(**kwargs)
