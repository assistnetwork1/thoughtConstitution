from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from constitution_engine.models.evidence import Evidence
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.types import InfoType
from constitution_engine.models.review import ReviewRecord


@dataclass(frozen=True)
class InvariantViolation:
    rule: str
    message: str


# ---------------------------
# v0.5.1 Types (bridge-safe)
# ---------------------------

class UncertaintyLevel(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"
    UNKNOWN = "unknown"


class ImpactLevel(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class ReversibilityLevel(str, Enum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


class ActionClass(str, Enum):
    PROBE = "probe"
    LIMITED = "limited"
    COMMIT = "commit"


class Riskiness(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


# ---------------------------
# Observations
# ---------------------------

def require_observations_are_observational(
    observations: Iterable[Observation],
) -> Sequence[InvariantViolation]:
    """
    Observations must be observational InfoTypes only.
    (Defensive: Observation.__post_init__ may already enforce this.)
    """
    violations: list[InvariantViolation] = []
    allowed = {InfoType.FACT, InfoType.MEASUREMENT, InfoType.EVENT, InfoType.TESTIMONY}

    for obs in observations:
        if obs.info_type not in allowed:
            violations.append(
                InvariantViolation(
                    rule="observation_type",
                    message=f"Observation {obs.observation_id} has non-observational type {obs.info_type}",
                )
            )
    return tuple(violations)


def require_observations_have_provenance(
    observations: Iterable[Observation],
) -> Sequence[InvariantViolation]:
    """
    Observation must not be provenance-empty:
      - raw_input_ids non-empty OR evidence_ids non-empty
    """
    violations: list[InvariantViolation] = []
    for obs in observations:
        if not obs.has_provenance():
            violations.append(
                InvariantViolation(
                    rule="observation_provenance",
                    message=(
                        f"Observation {obs.observation_id} has no provenance "
                        "(raw_input_ids and evidence_ids are both empty)."
                    ),
                )
            )
    return tuple(violations)


def require_observations_reference_existing_evidence(
    observations: Iterable[Observation],
    evidence_by_id: Mapping[str, Evidence],
) -> Sequence[InvariantViolation]:
    """
    If obs.evidence_ids is non-empty, every referenced Evidence id must exist.
    """
    violations: list[InvariantViolation] = []
    for obs in observations:
        if not obs.evidence_ids:
            continue
        missing = [eid for eid in obs.evidence_ids if eid not in evidence_by_id]
        if missing:
            violations.append(
                InvariantViolation(
                    rule="observation_evidence_link",
                    message=f"Observation {obs.observation_id} references missing Evidence IDs: {', '.join(missing)}",
                )
            )
    return tuple(violations)


# ---------------------------
# Evidence
# ---------------------------

def require_evidence_has_sources(
    evidence_items: Iterable[Evidence],
) -> Sequence[InvariantViolation]:
    """
    Evidence must have at least one source.
    """
    violations: list[InvariantViolation] = []
    for ev in evidence_items:
        if not ev.sources:
            violations.append(
                InvariantViolation(
                    rule="evidence_sources",
                    message=f"Evidence {ev.evidence_id} has no sources.",
                )
            )
    return tuple(violations)


def require_evidence_source_uris_nonempty(
    evidence_items: Iterable[Evidence],
) -> Sequence[InvariantViolation]:
    """
    Evidence sources must have non-empty URIs.
    """
    violations: list[InvariantViolation] = []
    for ev in evidence_items:
        for idx, src in enumerate(ev.sources):
            if not src.uri or not src.uri.strip():
                violations.append(
                    InvariantViolation(
                        rule="evidence_source_uri",
                        message=f"Evidence {ev.evidence_id} source[{idx}] has empty uri.",
                    )
                )
    return tuple(violations)


# ---------------------------
# Recommendations
# ---------------------------

def require_recommendation_has_orientation(rec: Recommendation) -> Sequence[InvariantViolation]:
    """
    Recommendation must have non-empty orientation_id.
    (Defensive: Recommendation.__post_init__ may already enforce this.)
    """
    if not rec.orientation_id:
        return (
            InvariantViolation(
                rule="recommendation_orientation",
                message="Recommendation missing orientation_id",
            ),
        )
    return tuple()


def require_recommendation_has_ranked_options(rec: Recommendation) -> Sequence[InvariantViolation]:
    """
    Recommendation must have at least one RankedOption.
    """
    if not rec.ranked_options:
        return (
            InvariantViolation(
                rule="recommendation_ranked_options",
                message="Recommendation has no ranked_options",
            ),
        )
    return tuple()


def require_recommendation_has_provenance(rec: Recommendation) -> Sequence[InvariantViolation]:
    """
    If a Recommendation ranks options, it must carry at least one provenance pointer:
      - evidence_ids OR observation_ids OR interpretation_ids OR model_state_ids
    """
    if not rec.ranked_options:
        return tuple()

    if not (rec.evidence_ids or rec.observation_ids or rec.interpretation_ids or rec.model_state_ids):
        return (
            InvariantViolation(
                rule="recommendation_provenance",
                message=(
                    "Recommendation has ranked_options but no provenance pointers "
                    "(evidence_ids, observation_ids, interpretation_ids, model_state_ids are all empty)."
                ),
            ),
        )
    return tuple()


def require_recommendation_references_existing_options(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
) -> Sequence[InvariantViolation]:
    """
    Every RankedOption.option_id referenced by the Recommendation must exist.
    """
    if not rec.ranked_options:
        return tuple()

    missing = [ro.option_id for ro in rec.ranked_options if ro.option_id not in options_by_id]
    if missing:
        return (
            InvariantViolation(
                rule="recommendation_option_link",
                message=f"Recommendation references missing Option IDs: {', '.join(missing)}",
            ),
        )
    return tuple()


def require_recommendation_provenance_overlaps_top_option(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
) -> Sequence[InvariantViolation]:
    """
    Recommendation provenance must connect to what it is recommending.

    If the Recommendation has a top-ranked Option, then the Recommendation must share
    at least one supporting reference with that Option:
      - overlap in observation_ids OR overlap in evidence_ids

    This prevents "provenance drift" (recommendation cites unrelated sources).
    """
    if not rec.ranked_options:
        return tuple()

    top_id = rec.top_option_id()
    if not top_id:
        return tuple()

    top = options_by_id.get(top_id)
    if top is None:
        # Missing refs handled elsewhere
        return tuple()

    rec_obs = set(rec.observation_ids)
    rec_ev = set(rec.evidence_ids)

    top_obs = set(top.observation_ids)
    top_ev = set(top.evidence_ids)

    has_obs_overlap = bool(rec_obs.intersection(top_obs)) if (rec_obs and top_obs) else False
    has_ev_overlap = bool(rec_ev.intersection(top_ev)) if (rec_ev and top_ev) else False

    if not (has_obs_overlap or has_ev_overlap):
        return (
            InvariantViolation(
                rule="recommendation_provenance_overlap_top_option",
                message=(
                    f"Recommendation {rec.recommendation_id} top option {top.option_id} has no provenance overlap: "
                    "no shared observation_ids or evidence_ids."
                ),
            ),
        )

    return tuple()


# ---------------------------
# EXECUTE option constraints
# ---------------------------

def require_execute_options_are_auditable_and_oriented(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
) -> Sequence[InvariantViolation]:
    """
    If a Recommendation ranks an EXECUTE option, that option must:
      - be auditable (have upstream references),
      - have option.orientation_id set,
      - match rec.orientation_id,
      - declare at least one uncertainty.
    """
    violations: list[InvariantViolation] = []

    for ro in rec.ranked_options:
        opt = options_by_id.get(ro.option_id)
        if opt is None:
            continue

        if opt.kind != OptionKind.EXECUTE:
            continue

        if not opt.has_upstream_references():
            violations.append(
                InvariantViolation(
                    rule="execute_option_auditability",
                    message=(
                        f"EXECUTE Option {opt.option_id} has no upstream references "
                        "(observation_ids / interpretation_ids / evidence_ids)."
                    ),
                )
            )

        if not opt.orientation_id:
            violations.append(
                InvariantViolation(
                    rule="execute_option_orientation",
                    message=f"EXECUTE Option {opt.option_id} missing orientation_id.",
                )
            )
        elif opt.orientation_id != rec.orientation_id:
            violations.append(
                InvariantViolation(
                    rule="execute_option_orientation_mismatch",
                    message=(
                        f"EXECUTE Option {opt.option_id} orientation_id={opt.orientation_id} "
                        f"does not match Recommendation orientation_id={rec.orientation_id}."
                    ),
                )
            )

        if not opt.uncertainties:
            violations.append(
                InvariantViolation(
                    rule="execute_option_uncertainty_required",
                    message=(
                        f"EXECUTE Option {opt.option_id} has no uncertainties; "
                        "execution requires explicit uncertainty."
                    ),
                )
            )

    return tuple(violations)


# ---------------------------
# v0.5.1 ActionClass Gate (canonical) — bridges from floats
# ---------------------------

def _max_uncertainty_float(opt: Option) -> float | None:
    """
    Returns max(u.level) if uncertainties exist; otherwise None.
    """
    if not getattr(opt, "uncertainties", None):
        return None
    levels = [u.level for u in opt.uncertainties]
    if not levels:
        return None
    return max(levels)


def bucket_uncertainty_level(opt: Option, *, high: float = 0.7, med: float = 0.3) -> UncertaintyLevel:
    """
    Bridge mapping: float -> ordinal UncertaintyLevel.
    - None => UNKNOWN
    - >= high => HIGH
    - >= med  => MED
    - else    => LOW
    """
    m = _max_uncertainty_float(opt)
    if m is None:
        return UncertaintyLevel.UNKNOWN
    if m >= high:
        return UncertaintyLevel.HIGH
    if m >= med:
        return UncertaintyLevel.MED
    return UncertaintyLevel.LOW


def bucket_impact_level(opt: Option, *, high: float = 0.7, med: float = 0.4) -> ImpactLevel:
    """
    Bridge mapping: opt.impact.value float -> ImpactLevel.
    """
    v = getattr(getattr(opt, "impact", None), "value", None)
    if v is None:
        # Treat missing impact as high-risk signal during bridge.
        return ImpactLevel.HIGH
    if v >= high:
        return ImpactLevel.HIGH
    if v >= med:
        return ImpactLevel.MED
    return ImpactLevel.LOW


def bucket_reversibility_level(opt: Option, *, low: float = 0.3, med: float = 0.6) -> ReversibilityLevel:
    """
    Bridge mapping: opt.reversibility.value float -> ReversibilityLevel.
    NOTE: reversibility LOW means hard to reverse.
    """
    v = getattr(getattr(opt, "reversibility", None), "value", None)
    if v is None:
        return ReversibilityLevel.LOW
    if v <= low:
        return ReversibilityLevel.LOW
    if v <= med:
        return ReversibilityLevel.MED
    return ReversibilityLevel.HIGH


def compute_riskiness(impact: ImpactLevel, reversibility: ReversibilityLevel) -> Riskiness:
    """
    Canonical table-ish version:
      - HIGH impact + LOW reversibility => HIGH riskiness
      - LOW impact + HIGH reversibility => LOW riskiness
      - otherwise => MED, with conservative elevation rules
    """
    if impact == ImpactLevel.HIGH and reversibility == ReversibilityLevel.LOW:
        return Riskiness.HIGH

    if impact == ImpactLevel.LOW and reversibility == ReversibilityLevel.HIGH:
        return Riskiness.LOW

    if impact == ImpactLevel.HIGH and reversibility in (ReversibilityLevel.MED, ReversibilityLevel.LOW):
        return Riskiness.HIGH

    if reversibility == ReversibilityLevel.LOW and impact in (ImpactLevel.MED, ImpactLevel.HIGH):
        return Riskiness.HIGH

    if impact == ImpactLevel.LOW and reversibility in (ReversibilityLevel.MED, ReversibilityLevel.HIGH):
        return Riskiness.LOW

    return Riskiness.MED


def allowed_action_classes(
    impact: ImpactLevel,
    reversibility: ReversibilityLevel,
    uncertainty: UncertaintyLevel,
) -> set[ActionClass]:
    """
    v0.5.1 canonical gate:
      - risk=HIGH => uncertainty must be LOW for LIMITED/COMMIT; else PROBE only
      - risk=MED  => uncertainty must be <= MED for LIMITED; else PROBE only
      - risk=LOW  => any uncertainty allowed (declared)
    """
    risk = compute_riskiness(impact, reversibility)

    if risk == Riskiness.HIGH:
        if uncertainty != UncertaintyLevel.LOW:
            return {ActionClass.PROBE}
        return {ActionClass.PROBE, ActionClass.LIMITED, ActionClass.COMMIT}

    if risk == Riskiness.MED:
        if uncertainty in (UncertaintyLevel.HIGH, UncertaintyLevel.UNKNOWN):
            return {ActionClass.PROBE}
        return {ActionClass.PROBE, ActionClass.LIMITED}

    return {ActionClass.PROBE, ActionClass.LIMITED, ActionClass.COMMIT}


def require_action_class_declared(opt: Option) -> Sequence[InvariantViolation]:
    """
    INV-ACT-001: Option must declare ActionClass.
    Bridge: accepts either ActionClass enum or a string in {"probe","limited","commit"}.
    """
    ac = getattr(opt, "action_class", None)
    if ac is None:
        return (
            InvariantViolation(
                rule="INV-ACT-001",
                message=f"Option {opt.option_id} missing action_class (required: probe/limited/commit).",
            ),
        )

    if isinstance(ac, ActionClass):
        return tuple()

    if isinstance(ac, str) and ac.strip().lower() in {a.value for a in ActionClass}:
        return tuple()

    return (
        InvariantViolation(
            rule="INV-ACT-001",
            message=(
                f"Option {opt.option_id} has invalid action_class={ac!r}. "
                "Must be one of: 'probe', 'limited', 'commit'."
            ),
        ),
    )


def _coerce_action_class(opt: Option) -> ActionClass | None:
    ac = getattr(opt, "action_class", None)
    if isinstance(ac, ActionClass):
        return ac
    if isinstance(ac, str):
        s = ac.strip().lower()
        for a in ActionClass:
            if a.value == s:
                return a
    return None


def require_proportionate_action_v051(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
) -> Sequence[InvariantViolation]:
    """
    v0.5.1 deterministic gate:
      - For EXECUTE options only, enforce:
        declared ActionClass ∈ allowed_action_classes(ImpactLevel, ReversibilityLevel, UncertaintyLevel)

    Override semantics (thin-slice):
      - If Recommendation.override_used=True AND "ALLOW_GATE_BYPASS" ∈ override_scope_used,
        then gate violations do NOT emit INV-ACT-002 (they are allowed but must be reviewed/audited).
    """
    violations: list[InvariantViolation] = []

    override_used = bool(getattr(rec, "override_used", False))
    override_scope_used = getattr(rec, "override_scope_used", ()) or ()
    scope_set = set(override_scope_used) if isinstance(override_scope_used, (list, tuple, set)) else set()
    gate_bypass = override_used and ("ALLOW_GATE_BYPASS" in scope_set)

    for ro in rec.ranked_options:
        opt = options_by_id.get(ro.option_id)
        if opt is None:
            continue
        if opt.kind != OptionKind.EXECUTE:
            continue

        violations.extend(require_action_class_declared(opt))
        ac = _coerce_action_class(opt)
        if ac is None:
            continue

        impact = bucket_impact_level(opt)
        reversibility = bucket_reversibility_level(opt)
        uncertainty = bucket_uncertainty_level(opt)

        allowed = allowed_action_classes(impact, reversibility, uncertainty)
        if ac not in allowed:
            if gate_bypass:
                continue

            violations.append(
                InvariantViolation(
                    rule="INV-ACT-002",
                    message=(
                        f"Option {opt.option_id} action_class={ac.value} not allowed by gate "
                        f"(impact={impact.value}, reversibility={reversibility.value}, uncertainty={uncertainty.value}). "
                        f"Allowed={sorted([a.value for a in allowed])}."
                    ),
                )
            )

    return tuple(violations)


# ---------------------------
# Proportionate action gate (legacy numeric heuristic) — optional
# ---------------------------

def require_proportionate_action_legacy_numeric(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
    *,
    high_uncertainty: float = 0.7,
    low_reversibility: float = 0.3,
    nontrivial_impact: float = 0.4,
) -> Sequence[InvariantViolation]:
    """
    Legacy heuristic gate (pre-v0.5 semantics). Keep it only temporarily.
    """
    violations: list[InvariantViolation] = []

    ranked_option_ids = [ro.option_id for ro in rec.ranked_options]
    ranked_options: list[Option] = []
    missing: list[str] = []

    for oid in ranked_option_ids:
        opt = options_by_id.get(oid)
        if opt is None:
            missing.append(oid)
        else:
            ranked_options.append(opt)

    if missing:
        violations.append(
            InvariantViolation(
                rule="proportionate_action_missing_options",
                message=f"Recommendation references missing Option IDs: {', '.join(missing)}",
            )
        )
        return tuple(violations)

    def max_unc(opt: Option) -> float:
        if not opt.uncertainties:
            return 0.0
        return max(u.level for u in opt.uncertainties)

    risky_execute_options = [
        opt for opt in ranked_options
        if (opt.kind == OptionKind.EXECUTE)
        and (max_unc(opt) >= high_uncertainty)
        and (opt.reversibility.value <= low_reversibility)
        and (opt.impact.value >= nontrivial_impact)
    ]

    if not risky_execute_options:
        return tuple()

    has_hedge_or_learn = any(
        opt.kind in {OptionKind.HEDGE, OptionKind.INFO_GATHERING}
        for opt in ranked_options
    )

    if not has_hedge_or_learn:
        witness = risky_execute_options[0]
        violations.append(
            InvariantViolation(
                rule="proportionate_action_legacy_numeric",
                message=(
                    "Risky EXECUTE option present (high uncertainty, low reversibility, non-trivial impact), "
                    f"but no HEDGE or INFO_GATHERING option included. Witness option_id={witness.option_id}."
                ),
            )
        )

    return tuple(violations)


# ---------------------------
# Public/stable gate entrypoint (validate.py import contract)
# ---------------------------

def require_proportionate_action(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
    *,
    use_legacy_numeric_gate: bool = False,
) -> Sequence[InvariantViolation]:
    """
    Public/stable entrypoint for proportionate action governance.

    validate.py historically imports `require_proportionate_action` from rules.py.
    In v0.5.1, constitutional governance is ActionClass gating (require_proportionate_action_v051).

    If `use_legacy_numeric_gate=True`, we also run the legacy numeric heuristic as an additional
    (optional) check during migration.
    """
    violations: list[InvariantViolation] = []
    violations.extend(require_proportionate_action_v051(rec, options_by_id))
    if use_legacy_numeric_gate:
        violations.extend(require_proportionate_action_legacy_numeric(rec, options_by_id))
    return tuple(violations)


# ---------------------------
# Orchestrators
# ---------------------------

def validate_observations(
    observations: Iterable[Observation],
    evidence_by_id: Mapping[str, Evidence],
) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []
    violations.extend(require_observations_are_observational(observations))
    violations.extend(require_observations_have_provenance(observations))
    violations.extend(require_observations_reference_existing_evidence(observations, evidence_by_id))
    return tuple(violations)


def validate_evidence(
    evidence_items: Iterable[Evidence],
) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []
    violations.extend(require_evidence_has_sources(evidence_items))
    violations.extend(require_evidence_source_uris_nonempty(evidence_items))
    return tuple(violations)


def validate_recommendation(
    rec: Recommendation,
    *,
    options_by_id: Mapping[str, Option],
    use_legacy_numeric_gate: bool = False,
) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []
    violations.extend(require_recommendation_has_orientation(rec))
    violations.extend(require_recommendation_has_ranked_options(rec))
    violations.extend(require_recommendation_has_provenance(rec))
    violations.extend(require_recommendation_references_existing_options(rec, options_by_id))
    violations.extend(require_execute_options_are_auditable_and_oriented(rec, options_by_id))

    violations.extend(
        require_proportionate_action(
            rec,
            options_by_id,
            use_legacy_numeric_gate=use_legacy_numeric_gate,
        )
    )

    violations.extend(require_recommendation_provenance_overlaps_top_option(rec, options_by_id))
    return tuple(violations)


def validate_all(
    *,
    observations: Iterable[Observation],
    evidence_items: Iterable[Evidence],
    options: Iterable[Option],
    recommendation: Recommendation,
    use_legacy_numeric_gate: bool = False,
) -> Sequence[InvariantViolation]:
    """
    One-call validation entrypoint.
    """
    evidence_by_id = {ev.evidence_id: ev for ev in evidence_items}
    options_by_id = {opt.option_id: opt for opt in options}

    violations: list[InvariantViolation] = []
    violations.extend(validate_evidence(evidence_items))
    violations.extend(validate_observations(observations, evidence_by_id))
    violations.extend(
        validate_recommendation(
            recommendation,
            options_by_id=options_by_id,
            use_legacy_numeric_gate=use_legacy_numeric_gate,
        )
    )
    return tuple(violations)


def require_review_exists_if_override_used(
    *,
    episode_id: str,
    recommendations: Iterable[Recommendation],
    review_ids: Sequence[str],
) -> Sequence[InvariantViolation]:
    """
    INV-REV-001:
      If any recommendation has override_used=True,
      then the episode must include at least one ReviewRecord id.
    """
    used = [r for r in recommendations if getattr(r, "override_used", False) is True]
    if not used:
        return tuple()

    if not review_ids:
        return (
            InvariantViolation(
                rule="INV-REV-001",
                message=(
                    f"DecisionEpisode {episode_id} contains override_used=true recommendations "
                    "but has no review_ids (override requires review)."
                ),
            ),
        )

    return tuple()


def require_review_audits_overrides(
    *,
    episode_id: str,
    recommendations: Iterable[Recommendation],
    review: ReviewRecord,
) -> Sequence[InvariantViolation]:
    """
    INV-REV-002:
      If overrides were used, the (latest) ReviewRecord must include an override audit entry
      per overridden recommendation.

    Minimal audit entry requirements (thin-slice):
      - recommendation_id
      - override_scope_used (list/tuple/set)
      - rationale (non-empty str)
    """
    overridden = [r for r in recommendations if getattr(r, "override_used", False) is True]
    if not overridden:
        return tuple()

    audit = review.override_audit or {}
    overrides = audit.get("overrides")

    if not isinstance(overrides, list):
        return (
            InvariantViolation(
                rule="INV-REV-002",
                message=(
                    f"DecisionEpisode {episode_id} override_used=true but ReviewRecord {review.review_id} "
                    "override_audit['overrides'] is missing or not a list."
                ),
            ),
        )

    # index audit entries by recommendation_id
    by_rec_id: dict[str, Mapping[str, object]] = {}
    for entry in overrides:
        if isinstance(entry, dict):
            rid = entry.get("recommendation_id")
            if isinstance(rid, str) and rid:
                by_rec_id[rid] = entry

    violations: list[InvariantViolation] = []

    for rec in overridden:
        rid = rec.recommendation_id
        entry = by_rec_id.get(rid)
        if entry is None:
            violations.append(
                InvariantViolation(
                    rule="INV-REV-002",
                    message=(
                        f"DecisionEpisode {episode_id} override_used=true for Recommendation {rid} "
                        f"but ReviewRecord {review.review_id} has no matching override audit entry."
                    ),
                )
            )
            continue

        scope_used = entry.get("override_scope_used")
        rationale = entry.get("rationale")

        scope_ok = isinstance(scope_used, (list, tuple, set)) and len(scope_used) > 0
        rationale_ok = isinstance(rationale, str) and bool(rationale.strip())

        if not scope_ok or not rationale_ok:
            violations.append(
                InvariantViolation(
                    rule="INV-REV-002",
                    message=(
                        f"ReviewRecord {review.review_id} override audit entry for Recommendation {rid} "
                        "is missing required fields (override_scope_used non-empty; rationale non-empty)."
                    ),
                )
            )

    return tuple(violations)
