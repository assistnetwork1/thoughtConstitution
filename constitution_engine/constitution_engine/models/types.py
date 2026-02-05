from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, NewType, Optional, Sequence, Tuple
from uuid import uuid4


# ============================================================
# types.py (kernel scalars + taxonomies)
# ============================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


# Information typing ONLY (not artifact typing)
class InfoType(str, Enum):
    # Observational (reality-anchored)
    FACT = "fact"
    MEASUREMENT = "measurement"
    EVENT = "event"
    TESTIMONY = "testimony"

    # Interpretive
    CLAIM = "claim"
    EXPLANATION = "explanation"
    HYPOTHESIS = "hypothesis"
    FRAME = "frame"

    # Normative
    VALUE = "value"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"

    # Predictive
    FORECAST = "forecast"
    SCENARIO = "scenario"


# Artifact typing (for storage/indexing, if needed)
class ArtifactType(str, Enum):
    RAW_INPUT = "RawInput"
    EVIDENCE = "Evidence"
    OBSERVATION = "Observation"
    INTERPRETATION = "Interpretation"
    MODEL_SPEC = "ModelSpec"
    MODEL_STATE = "ModelState"
    ORIENTATION = "Orientation"
    OPTION = "Option"
    RECOMMENDATION = "Recommendation"
    OUTCOME = "Outcome"
    REVIEW = "ReviewRecord"
    AUDIT = "AuditTrail"
    EPISODE = "DecisionEpisode"


@dataclass(frozen=True)
class Confidence:
    """
    Epistemic confidence: how well-supported something is by evidence.
    Range: 0.0–1.0
    """
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Confidence.value must be between 0.0 and 1.0")


class UncertaintyKind(str, Enum):
    MISSING_DATA = "missing_data"
    AMBIGUITY = "ambiguity"
    VARIANCE = "variance"
    MODEL_ERROR = "model_error"
    ADVERSARIAL = "adversarial"
    OTHER = "other"


@dataclass(frozen=True)
class Uncertainty:
    """
    Explicit uncertainty.
    - Confidence: support strength
    - Uncertainty: remaining unknowns/ambiguity/variance even if supported
    Range: 0.0–1.0
    """
    description: str
    level: float  # 0.0–1.0
    kind: UncertaintyKind = UncertaintyKind.OTHER

    def __post_init__(self) -> None:
        if not (0.0 <= self.level <= 1.0):
            raise ValueError("Uncertainty.level must be between 0.0 and 1.0")


@dataclass(frozen=True)
class Reversibility:
    """
    How reversible an action is. Higher means easier to undo.
    Range: 0.0–1.0
    """
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Reversibility.value must be between 0.0 and 1.0")


@dataclass(frozen=True)
class Impact:
    """
    Magnitude of downside/upside exposure if executed.
    Range: 0.0–1.0 (unitless kernel-scale; apps can map to dollars/time/etc.)
    """
    value: float
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Impact.value must be between 0.0 and 1.0")


class RiskPosture(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class Strength:
    """
    Recommendation push intensity (normatively consequential).
    Range: 0.0–1.0
    """
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Strength.value must be between 0.0 and 1.0")


@dataclass(frozen=True)
class Score:
    """
    Internal ranking signal (non-normative).
    Optional: useful to keep ranking separate from Strength.
    Range: 0.0–1.0
    """
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Score.value must be between 0.0 and 1.0")


Weight = NewType("Weight", float)


# ============================================================
# Kernel artifacts (minimal, for end-to-end thin slice)
# ============================================================

@dataclass(frozen=True)
class RawInput:
    id: str
    created_at: datetime
    source_system: str
    payload: dict


@dataclass(frozen=True)
class Evidence:
    id: str
    raw_input_id: str
    retrieved_at: datetime
    locator: str
    span: Optional[str] = None
    content_hash: Optional[str] = None



@dataclass(frozen=True)
class Observation:
    id: str
    created_at: datetime
    info_type: InfoType  # must be observational
    statement: str
    confidence: Confidence
    uncertainty: Uncertainty
    evidence_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Interpretation:
    id: str
    created_at: datetime
    info_type: InfoType  # must be interpretive
    hypothesis: str
    assumptions: Tuple[str, ...]
    confidence: Confidence
    uncertainty: Uncertainty
    observation_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Orientation:
    id: str
    created_at: datetime
    objectives: Tuple[str, ...]
    constraints: Tuple[str, ...]
    weights: Dict[str, Weight]
    risk_posture: RiskPosture


@dataclass(frozen=True)
class Option:
    id: str
    created_at: datetime
    label: str
    description: str
    impact: Impact
    reversibility: Reversibility
    preconditions: Tuple[str, ...] = ()
    uncertainty: Uncertainty = field(default_factory=lambda: Uncertainty("unspecified", 0.5))


@dataclass(frozen=True)
class RankedOption:
    option_id: str
    rank: int
    strength: Strength
    score: Optional[Score] = None  # optional; internal ranking-only signal


@dataclass(frozen=True)
class Recommendation:
    id: str
    created_at: datetime
    confidence: Confidence
    uncertainty: Uncertainty
    rationale: str
    ranked_options: Tuple[RankedOption, ...]
    audit_trail_id: str


@dataclass(frozen=True)
class AuditLink:
    from_type: ArtifactType
    from_id: str
    rel: str
    to_type: ArtifactType
    to_id: str


@dataclass(frozen=True)
class AuditTrail:
    id: str
    created_at: datetime
    links: Tuple[AuditLink, ...]


@dataclass(frozen=True)
class DecisionEpisode:
    id: str
    created_at: datetime
    raw_input_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    observation_ids: Tuple[str, ...] = ()
    interpretation_ids: Tuple[str, ...] = ()
    orientation_id: Optional[str] = None
    option_ids: Tuple[str, ...] = ()
    recommendation_ids: Tuple[str, ...] = ()


# ============================================================
# In-memory store (reference impl)
# ============================================================

class ArtifactStore:
    def put(self, obj) -> None:
        raise NotImplementedError

    def get_episode(self, episode_id: str) -> DecisionEpisode:
        raise NotImplementedError

    def get_orientation(self, orientation_id: str) -> Orientation:
        raise NotImplementedError

    def get_raw_input(self, raw_input_id: str) -> RawInput:
        raise NotImplementedError

    def get_evidence(self, evidence_id: str) -> Evidence:
        raise NotImplementedError

    def get_observation(self, observation_id: str) -> Observation:
        raise NotImplementedError

    def get_interpretation(self, interpretation_id: str) -> Interpretation:
        raise NotImplementedError

    def get_option(self, option_id: str) -> Option:
        raise NotImplementedError

    def get_recommendation(self, recommendation_id: str) -> Recommendation:
        raise NotImplementedError

    def get_audit_trail(self, audit_trail_id: str) -> AuditTrail:
        raise NotImplementedError


class InMemoryStore(ArtifactStore):
    def __init__(self) -> None:
        self.raw_inputs: Dict[str, RawInput] = {}
        self.evidence: Dict[str, Evidence] = {}
        self.observations: Dict[str, Observation] = {}
        self.interpretations: Dict[str, Interpretation] = {}
        self.orientations: Dict[str, Orientation] = {}
        self.options: Dict[str, Option] = {}
        self.recommendations: Dict[str, Recommendation] = {}
        self.audit_trails: Dict[str, AuditTrail] = {}
        self.episodes: Dict[str, DecisionEpisode] = {}

    def put(self, obj) -> None:
        if isinstance(obj, RawInput):
            self.raw_inputs[obj.id] = obj
        elif isinstance(obj, Evidence):
            self.evidence[obj.id] = obj
        elif isinstance(obj, Observation):
            self.observations[obj.id] = obj
        elif isinstance(obj, Interpretation):
            self.interpretations[obj.id] = obj
        elif isinstance(obj, Orientation):
            self.orientations[obj.id] = obj
        elif isinstance(obj, Option):
            self.options[obj.id] = obj
        elif isinstance(obj, Recommendation):
            self.recommendations[obj.id] = obj
        elif isinstance(obj, AuditTrail):
            self.audit_trails[obj.id] = obj
        elif isinstance(obj, DecisionEpisode):
            self.episodes[obj.id] = obj
        else:
            raise TypeError(f"Unknown artifact type: {type(obj)}")

    def get_episode(self, episode_id: str) -> DecisionEpisode:
        return self.episodes[episode_id]

    def get_orientation(self, orientation_id: str) -> Orientation:
        return self.orientations[orientation_id]

    def get_raw_input(self, raw_input_id: str) -> RawInput:
        return self.raw_inputs[raw_input_id]

    def get_evidence(self, evidence_id: str) -> Evidence:
        return self.evidence[evidence_id]

    def get_observation(self, observation_id: str) -> Observation:
        return self.observations[observation_id]

    def get_interpretation(self, interpretation_id: str) -> Interpretation:
        return self.interpretations[interpretation_id]

    def get_option(self, option_id: str) -> Option:
        return self.options[option_id]

    def get_recommendation(self, recommendation_id: str) -> Recommendation:
        return self.recommendations[recommendation_id]

    def get_audit_trail(self, audit_trail_id: str) -> AuditTrail:
        return self.audit_trails[audit_trail_id]


# ============================================================
# Invariants + validation harness
# ============================================================

@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    artifact_id: Optional[str] = None


Invariant = callable


def validate_episode(store: ArtifactStore, episode_id: str, invariants: Iterable) -> List[Violation]:
    out: List[Violation] = []
    for inv in invariants:
        out.extend(list(inv(store, episode_id)))
    return out


OBS_OK = {InfoType.FACT, InfoType.MEASUREMENT, InfoType.EVENT, InfoType.TESTIMONY}
INT_OK = {InfoType.CLAIM, InfoType.EXPLANATION, InfoType.HYPOTHESIS, InfoType.FRAME}


def inv_episode_requires_orientation(store: ArtifactStore, episode_id: str) -> Sequence[Violation]:
    ep = store.get_episode(episode_id)
    if ep.orientation_id is None:
        return [Violation(
            code="orientation.missing",
            message="DecisionEpisode must include an Orientation before any Recommendation is valid.",
            artifact_id=episode_id,
        )]
    return []


def inv_recommendation_requires_audit(store: ArtifactStore, episode_id: str) -> Sequence[Violation]:
    ep = store.get_episode(episode_id)
    v: List[Violation] = []
    for rec_id in ep.recommendation_ids:
        rec = store.get_recommendation(rec_id)
        if not rec.audit_trail_id:
            v.append(Violation(
                code="audit.missing",
                message="Recommendation must include an audit_trail_id.",
                artifact_id=rec_id,
            ))
    return v


def inv_info_type_slots(store: ArtifactStore, episode_id: str) -> Sequence[Violation]:
    ep = store.get_episode(episode_id)
    v: List[Violation] = []

    for obs_id in ep.observation_ids:
        obs = store.get_observation(obs_id)
        if obs.info_type not in OBS_OK:
            v.append(Violation(
                code="infotype.observation.invalid",
                message=f"Observation must use observational InfoType; got {obs.info_type}.",
                artifact_id=obs_id,
            ))

    for it_id in ep.interpretation_ids:
        it = store.get_interpretation(it_id)
        if it.info_type not in INT_OK:
            v.append(Violation(
                code="infotype.interpretation.invalid",
                message=f"Interpretation must use interpretive InfoType; got {it.info_type}.",
                artifact_id=it_id,
            ))
    return v


def max_strength_allowed(
    *,
    confidence: Confidence,
    uncertainty: Uncertainty,
    impact: Impact,
    reversibility: Reversibility,
    risk_posture: RiskPosture,
) -> float:
    """
    Constitutional cap for recommendation strength in [0,1].
    - decreases as impact↑, irreversibility↑, uncertainty↑, confidence↓
    - risk posture shifts slightly, bounded
    """
    risk = 0.5 * impact.value + 0.5 * (1.0 - reversibility.value)
    epistemic = 0.6 * (1.0 - confidence.value) + 0.4 * uncertainty.level

    posture_adj = {
        RiskPosture.CONSERVATIVE: -0.10,
        RiskPosture.BALANCED: 0.00,
        RiskPosture.AGGRESSIVE: 0.10,
    }[risk_posture]

    raw = 1.0 - (0.95 * risk + 0.95 * epistemic) + posture_adj
    return max(0.0, min(1.0, raw))


def inv_proportionate_action(store: ArtifactStore, episode_id: str) -> Sequence[Violation]:
    ep = store.get_episode(episode_id)
    if ep.orientation_id is None:
        return []  # other invariant catches
    ori = store.get_orientation(ep.orientation_id)

    v: List[Violation] = []
    for rec_id in ep.recommendation_ids:
        rec = store.get_recommendation(rec_id)
        for ranked in rec.ranked_options:
            opt = store.get_option(ranked.option_id)
            cap = max_strength_allowed(
                confidence=rec.confidence,
                uncertainty=rec.uncertainty,
                impact=opt.impact,
                reversibility=opt.reversibility,
                risk_posture=ori.risk_posture,
            )
            if ranked.strength.value > cap + 1e-12:
                v.append(Violation(
                    code="action.disproportionate",
                    message=(
                        f"Strength {ranked.strength.value:.3f} exceeds cap {cap:.3f} "
                        f"(impact={opt.impact.value:.2f}, rev={opt.reversibility.value:.2f}, "
                        f"conf={rec.confidence.value:.2f}, unc={rec.uncertainty.level:.2f}, posture={ori.risk_posture})."
                    ),
                    artifact_id=rec_id,
                ))
    return v


def inv_traceability_minimum(store: ArtifactStore, episode_id: str) -> Sequence[Violation]:
    """
    Minimal auditability:
    - If there's a Recommendation, there must be at least one Observation or Interpretation or Evidence.
    - (Domain apps can tighten this later.)
    """
    ep = store.get_episode(episode_id)
    if not ep.recommendation_ids:
        return []
    if not (ep.observation_ids or ep.interpretation_ids or ep.evidence_ids):
        return [Violation(
            code="traceability.empty",
            message="Recommendations exist but no Observation/Interpretation/Evidence is recorded in the episode.",
            artifact_id=episode_id,
        )]
    return []


# ============================================================
# Audit dump
# ============================================================

def print_audit(store: ArtifactStore, audit_trail_id: str) -> None:
    trail = store.get_audit_trail(audit_trail_id)
    print(f"\nAuditTrail {trail.id} @ {trail.created_at.isoformat()}")
    for link in trail.links:
        print(f"  {link.from_type.value}:{link.from_id} --{link.rel}--> {link.to_type.value}:{link.to_id}")


# ============================================================
# Thin-slice toy episode builder (Observe → Model → Orient → Act)
# ============================================================

def build_toy_episode(store: InMemoryStore) -> str:
    # Episode container
    ep_id = new_id("ep")
    ep = DecisionEpisode(id=ep_id, created_at=now_utc())
    store.put(ep)

    # Raw input (telemetry)
    ri = RawInput(
        id=new_id("ri"),
        created_at=now_utc(),
        source_system="toy.telemetry",
        payload={"user_id": "u_123", "active_now": True, "session_seconds": 120},
    )
    store.put(ri)

    ev = Evidence(
        id=new_id("ev"),
        raw_input_id=ri.id,
        retrieved_at=now_utc(),
        locator="telemetry://toy/session/u_123",
        span="active_now, session_seconds",
        content_hash=None,
    )
    store.put(ev)

    # Observation (reality-anchored)
    obs = Observation(
        id=new_id("obs"),
        created_at=now_utc(),
        info_type=InfoType.MEASUREMENT,
        statement="User is active in-app (session_seconds>=120).",
        confidence=Confidence(0.75),
        uncertainty=Uncertainty("telemetry may be delayed or noisy", 0.25, UncertaintyKind.VARIANCE),
        evidence_ids=(ev.id,),
    )
    store.put(obs)

    # Interpretation (hypothesis)
    it = Interpretation(
        id=new_id("it"),
        created_at=now_utc(),
        info_type=InfoType.HYPOTHESIS,
        hypothesis="A notification sent now is likely to be seen.",
        assumptions=(
            "Current activity implies attention to the device",
            "Notification channel is enabled",
        ),
        confidence=Confidence(0.60),
        uncertainty=Uncertainty("attention and channel status uncertain", 0.45, UncertaintyKind.AMBIGUITY),
        observation_ids=(obs.id,),
    )
    store.put(it)

    # Orientation (values/constraints/risk)
    ori = Orientation(
        id=new_id("ori"),
        created_at=now_utc(),
        objectives=("Increase engagement", "Deliver timely information"),
        constraints=("Avoid annoyance", "Do not interrupt during focus states"),
        weights={"engagement": Weight(0.6), "annoyance": Weight(0.4)},
        risk_posture=RiskPosture.BALANCED,
    )
    store.put(ori)

    # Options
    opt_send = Option(
        id=new_id("opt"),
        created_at=now_utc(),
        label="Send now",
        description="Send notification immediately.",
        impact=Impact(0.55, "may annoy if mistimed; may boost engagement"),
        reversibility=Reversibility(0.20),
        preconditions=("Notification channel enabled",),
        uncertainty=Uncertainty("user context unknown", 0.55, UncertaintyKind.MISSING_DATA),
    )
    opt_delay = Option(
        id=new_id("opt"),
        created_at=now_utc(),
        label="Delay 1 hour",
        description="Schedule notification for ~1 hour later.",
        impact=Impact(0.30, "lower annoyance risk; may reduce timeliness"),
        reversibility=Reversibility(0.85),
        preconditions=("Notification channel enabled",),
        uncertainty=Uncertainty("future availability unknown", 0.50, UncertaintyKind.AMBIGUITY),
    )
    opt_skip = Option(
        id=new_id("opt"),
        created_at=now_utc(),
        label="Do not send",
        description="Skip sending the notification.",
        impact=Impact(0.20, "missed engagement opportunity; avoids annoyance"),
        reversibility=Reversibility(0.95),
        preconditions=(),
        uncertainty=Uncertainty("value of message unknown", 0.40, UncertaintyKind.OTHER),
    )
    store.put(opt_send)
    store.put(opt_delay)
    store.put(opt_skip)

    # Build audit trail links (lossless lineage edges)
    audit_id = new_id("audit")
    links: List[AuditLink] = []
    links.append(AuditLink(ArtifactType.EPISODE, ep_id, "contains", ArtifactType.RAW_INPUT, ri.id))
    links.append(AuditLink(ArtifactType.RAW_INPUT, ri.id, "supports", ArtifactType.EVIDENCE, ev.id))
    links.append(AuditLink(ArtifactType.EVIDENCE, ev.id, "supports", ArtifactType.OBSERVATION, obs.id))
    links.append(AuditLink(ArtifactType.OBSERVATION, obs.id, "informs", ArtifactType.INTERPRETATION, it.id))
    links.append(AuditLink(ArtifactType.INTERPRETATION, it.id, "informs", ArtifactType.ORIENTATION, ori.id))
    links.append(AuditLink(ArtifactType.ORIENTATION, ori.id, "governs", ArtifactType.OPTION, opt_send.id))
    links.append(AuditLink(ArtifactType.ORIENTATION, ori.id, "governs", ArtifactType.OPTION, opt_delay.id))
    links.append(AuditLink(ArtifactType.ORIENTATION, ori.id, "governs", ArtifactType.OPTION, opt_skip.id))

    audit = AuditTrail(id=audit_id, created_at=now_utc(), links=tuple(links))
    store.put(audit)

    # Act: produce a recommendation (toy ranking + then constitutional cap)
    # We intentionally keep "score" separate from "strength".
    # Rank: Delay > Send now > Skip (toy)
    rec_conf = Confidence(0.58)
    rec_unc = Uncertainty("limited context about user state", 0.55, UncertaintyKind.MISSING_DATA)

    ranked = [
        RankedOption(option_id=opt_delay.id, rank=1, score=Score(0.72), strength=Strength(0.80)),
        RankedOption(option_id=opt_send.id, rank=2, score=Score(0.65), strength=Strength(0.75)),
        RankedOption(option_id=opt_skip.id, rank=3, score=Score(0.40), strength=Strength(0.35)),
    ]

    # Apply constitutional caps immediately (so invariants can still catch mistakes)
    capped_ranked: List[RankedOption] = []
    for r in ranked:
        opt = store.get_option(r.option_id)
        cap = max_strength_allowed(
            confidence=rec_conf,
            uncertainty=rec_unc,
            impact=opt.impact,
            reversibility=opt.reversibility,
            risk_posture=ori.risk_posture,
        )
        capped_ranked.append(RankedOption(
            option_id=r.option_id,
            rank=r.rank,
            score=r.score,
            strength=Strength(min(r.strength.value, cap)),
        ))

    rec = Recommendation(
        id=new_id("rec"),
        created_at=now_utc(),
        confidence=rec_conf,
        uncertainty=rec_unc,
        rationale=(
            "Delay is preferred given uncertainty about user context and the annoyance constraint; "
            "sending now remains a viable fallback if timeliness dominates."
        ),
        ranked_options=tuple(capped_ranked),
        audit_trail_id=audit_id,
    )
    store.put(rec)

    # Update episode index (index-only container)
    ep2 = DecisionEpisode(
        id=ep.id,
        created_at=ep.created_at,
        raw_input_ids=(ri.id,),
        evidence_ids=(ev.id,),
        observation_ids=(obs.id,),
        interpretation_ids=(it.id,),
        orientation_id=ori.id,
        option_ids=(opt_send.id, opt_delay.id, opt_skip.id),
        recommendation_ids=(rec.id,),
    )
    store.put(ep2)

    return ep2.id


# ============================================================
# Runner
# ============================================================

def main() -> None:
    store = InMemoryStore()
    episode_id = build_toy_episode(store)

    invariants = [
        inv_episode_requires_orientation,
        inv_recommendation_requires_audit,
        inv_info_type_slots,
        inv_traceability_minimum,
        inv_proportionate_action,
    ]

    violations = validate_episode(store, episode_id, invariants)
    ep = store.get_episode(episode_id)

    print(f"DecisionEpisode {ep.id} @ {ep.created_at.isoformat()}")
    print(f"  raw_inputs: {len(ep.raw_input_ids)} | evidence: {len(ep.evidence_ids)} | "
          f"observations: {len(ep.observation_ids)} | interpretations: {len(ep.interpretation_ids)} | "
          f"options: {len(ep.option_ids)} | recommendations: {len(ep.recommendation_ids)}")

    if violations:
        print("\nVALIDATION: FAILED")
        for v in violations:
            print(f"- [{v.code}] {v.message} (artifact={v.artifact_id})")
        return

    print("\nVALIDATION: OK")

    rec = store.get_recommendation(ep.recommendation_ids[0])
    print(f"\nRecommendation {rec.id} @ {rec.created_at.isoformat()}")
    print(f"  confidence={rec.confidence.value:.2f} uncertainty={rec.uncertainty.level:.2f} ({rec.uncertainty.kind})")
    print(f"  rationale: {rec.rationale}")

    ori = store.get_orientation(ep.orientation_id)  # type: ignore[arg-type]
    for r in sorted(rec.ranked_options, key=lambda x: x.rank):
        opt = store.get_option(r.option_id)
        cap = max_strength_allowed(
            confidence=rec.confidence,
            uncertainty=rec.uncertainty,
            impact=opt.impact,
            reversibility=opt.reversibility,
            risk_posture=ori.risk_posture,
        )
        score_str = f"{r.score.value:.2f}" if r.score else "—"
        print(
            f"  #{r.rank}: {opt.label:12s} "
            f"(strength={r.strength.value:.2f} cap={cap:.2f} score={score_str}) "
            f"impact={opt.impact.value:.2f} rev={opt.reversibility.value:.2f}"
        )

    print_audit(store, rec.audit_trail_id)


if __name__ == "__main__":
    main()
