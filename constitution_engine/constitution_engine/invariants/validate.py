from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from constitution_engine.models.evidence import Evidence
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option
from constitution_engine.models.outcome import Outcome
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.review import ReviewRecord
from constitution_engine.runtime.store import ArtifactStore, ResolveError

from constitution_engine.models.calibration import CalibrationNote
from constitution_engine.models.choice import ChoiceRecord

from .rules import (
    InvariantViolation,
    require_choice_exists_if_episode_acted,
    validate_choices,
    require_observations_are_observational,
    require_proportionate_action,
    require_recommendation_has_orientation,
    require_recommendation_has_ranked_options,
    require_review_audits_overrides,
    require_review_exists_if_override_used,
    validate_all as _validate_all,  # re-export contract for tests expecting validate_all here
)


@dataclass(frozen=True)
class ValidationReport:
    """
    Collected violations and resolution errors for an artifact or episode.
    """
    subject: str
    violations: Sequence[InvariantViolation]
    resolve_errors: Sequence[ResolveError]

    @property
    def ok(self) -> bool:
        return (not self.violations) and (not self.resolve_errors)


def _as_missing_violations(
    errors: Sequence[ResolveError],
    *,
    context: str | None = None,
) -> Sequence[InvariantViolation]:
    prefix = f"{context}: " if context else ""
    return tuple(
        InvariantViolation(
            rule="missing_reference",
            message=f"{prefix}{e.artifact_type} missing: {e.artifact_id}",
        )
        for e in errors
    )


# ---------------------------
# Outcome invariants (episode-level)
# ---------------------------

def _episode_acted(ep: DecisionEpisode) -> bool:
    """
    v0.5.2 thin-slice: prefer the explicit field (ep.acted), but remain backward-compatible
    with any legacy "meta['acted']" usage.
    """
    try:
        if bool(getattr(ep, "acted", False)):
            return True
    except Exception:
        pass

    meta = getattr(ep, "meta", {}) or {}
    try:
        return bool(meta.get("acted", False))
    except Exception:
        return False


def require_outcome_exists_if_episode_acted(
    *,
    episode_id: str,
    acted: bool,
    has_recommendation: bool,
    outcome_ids: Sequence[str],
) -> Sequence[InvariantViolation]:
    """
    INV-OUT-001:
      If the episode has a recommendation AND the user/system marked it as acted,
      then at least one outcome_id must exist.
    """
    if not acted:
        return tuple()
    if not has_recommendation:
        return tuple()
    if outcome_ids:
        return tuple()

    return (
        InvariantViolation(
            rule="INV-OUT-001",
            message=(
                f"DecisionEpisode {episode_id} is marked acted=true and has recommendation_ids, "
                "but has no outcome_ids (acting requires an Outcome)."
            ),
        ),
    )


def validate_outcomes(
    *,
    outcomes: Sequence[Outcome],
    recommendations_by_id: Mapping[str, Recommendation],
    options_by_id: Mapping[str, Option],
) -> Sequence[InvariantViolation]:
    """
    Minimal thin-slice checks:
      - If outcome.recommendation_id is set, it must exist
      - If outcome.chosen_option_id is set, it must exist
      - If both are set, chosen option should be one of the recommendation's ranked option ids
        (soft guardrail for coherence)
    """
    violations: list[InvariantViolation] = []

    for out in outcomes:
        rid = out.recommendation_id
        oid = out.chosen_option_id

        if rid and rid not in recommendations_by_id:
            violations.append(
                InvariantViolation(
                    rule="INV-OUT-002",
                    message=f"Outcome {out.outcome_id} references missing Recommendation {rid}.",
                )
            )

        if oid and oid not in options_by_id:
            violations.append(
                InvariantViolation(
                    rule="INV-OUT-003",
                    message=f"Outcome {out.outcome_id} references missing Option {oid}.",
                )
            )

        if rid and oid and (rid in recommendations_by_id):
            rec = recommendations_by_id[rid]
            ranked_ids = {ro.option_id for ro in rec.ranked_options}
            if ranked_ids and (oid not in ranked_ids):
                violations.append(
                    InvariantViolation(
                        rule="INV-OUT-004",
                        message=(
                            f"Outcome {out.outcome_id} chosen_option_id={oid} is not present in "
                            f"Recommendation {rid} ranked_options."
                        ),
                    )
                )

    return tuple(violations)


# ---------------------------
# Calibration invariants (episode-level)
# ---------------------------

def validate_calibrations(
    *,
    calibrations: Sequence[CalibrationNote],
    episode: DecisionEpisode,
    reviews_by_id: Mapping[str, ReviewRecord],
    outcomes_by_id: Mapping[str, Outcome],
) -> Sequence[InvariantViolation]:
    """
    Thin-slice:
      - INV-CAL-001: calibration.episode_id must match the episode being validated
      - INV-CAL-002: calibration.review_id must be present
      - INV-CAL-003: calibration.review_id must exist
      - INV-CAL-004: each calibration.outcome_id must exist (if provided)
    """
    violations: list[InvariantViolation] = []

    for cal in calibrations:
        # 1) episode_id must match
        if getattr(cal, "episode_id", "") and (cal.episode_id != episode.episode_id):
            violations.append(
                InvariantViolation(
                    rule="INV-CAL-001",
                    message=(
                        f"CalibrationNote {cal.calibration_id} episode_id={cal.episode_id} "
                        f"does not match DecisionEpisode {episode.episode_id}."
                    ),
                )
            )

        # 2) review_id must be present and exist
        rid = getattr(cal, "review_id", None)
        if not rid:
            violations.append(
                InvariantViolation(
                    rule="INV-CAL-002",
                    message=f"CalibrationNote {cal.calibration_id} missing review_id.",
                )
            )
        elif rid not in reviews_by_id:
            violations.append(
                InvariantViolation(
                    rule="INV-CAL-003",
                    message=f"CalibrationNote {cal.calibration_id} references missing ReviewRecord {rid}.",
                )
            )

        # 3) outcome_ids must exist (if provided)
        for oid in tuple(getattr(cal, "outcome_ids", tuple()) or tuple()):
            if oid and oid not in outcomes_by_id:
                violations.append(
                    InvariantViolation(
                        rule="INV-CAL-004",
                        message=f"CalibrationNote {cal.calibration_id} references missing Outcome {oid}.",
                    )
                )

    return tuple(violations)


# ---------------------------
# Functional validation API (import contract)
# ---------------------------

def validate_all(
    *,
    observations: Iterable[Observation],
    evidence_items: Iterable[Evidence],
    options: Iterable[Option],
    recommendation: Recommendation,
    use_legacy_numeric_gate: bool = False,
) -> Sequence[InvariantViolation]:
    """
    Pure functional entrypoint (no ArtifactStore). Kept for backwards-compatibility with tests.

    Delegates to invariants.rules.validate_all (canonical).
    """
    return _validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=options,
        recommendation=recommendation,
        use_legacy_numeric_gate=use_legacy_numeric_gate,
    )


# ---------------------------
# ArtifactStore-based validators
# ---------------------------

def validate_recommendation(
    store: ArtifactStore,
    rec_id: str,
) -> ValidationReport:
    violations: list[InvariantViolation] = []
    resolve_errors: list[ResolveError] = []

    # Resolve Recommendation (don't throw)
    try:
        rec = store.must_get(Recommendation, rec_id)
    except ResolveError as err:
        resolve_errors.append(err)
        violations.append(
            InvariantViolation(rule="missing_reference", message=f"Recommendation missing: {rec_id}")
        )
        return ValidationReport(
            subject=f"Recommendation:{rec_id}",
            violations=tuple(violations),
            resolve_errors=tuple(resolve_errors),
        )

    # Basic invariants
    violations.extend(require_recommendation_has_orientation(rec))
    violations.extend(require_recommendation_has_ranked_options(rec))

    # Resolve referenced Options for proportionate action
    opt_ids = [ro.option_id for ro in rec.ranked_options]
    options, opt_errors = store.resolve_many(Option, opt_ids)
    resolve_errors.extend(opt_errors)

    options_by_id: Mapping[str, Option] = {o.option_id: o for o in options}
    if opt_errors:
        violations.extend(_as_missing_violations(opt_errors, context=f"Recommendation:{rec_id}"))
    else:
        # canonical v0.5.1 gate
        violations.extend(require_proportionate_action(rec, options_by_id))

    return ValidationReport(
        subject=f"Recommendation:{rec_id}",
        violations=tuple(violations),
        resolve_errors=tuple(resolve_errors),
    )


def validate_episode(
    store: ArtifactStore,
    episode_id: str,
) -> ValidationReport:
    violations: list[InvariantViolation] = []
    resolve_errors: list[ResolveError] = []

    # Resolve Episode (don't throw)
    try:
        ep = store.must_get(DecisionEpisode, episode_id)
    except ResolveError as err:
        resolve_errors.append(err)
        violations.append(
            InvariantViolation(rule="missing_reference", message=f"DecisionEpisode missing: {episode_id}")
        )
        return ValidationReport(
            subject=f"DecisionEpisode:{episode_id}",
            violations=tuple(violations),
            resolve_errors=tuple(resolve_errors),
        )

    # Resolve observations and validate observational purity
    observations, obs_errors = store.resolve_many(Observation, ep.observation_ids)
    resolve_errors.extend(obs_errors)
    if obs_errors:
        violations.extend(_as_missing_violations(obs_errors, context=f"DecisionEpisode:{episode_id}"))
    else:
        violations.extend(require_observations_are_observational(observations))

    # Resolve episode options once (used by Outcome checks and potentially other rules)
    opts, opt_errors = store.resolve_many(Option, ep.option_ids)
    resolve_errors.extend(opt_errors)
    if opt_errors:
        violations.extend(_as_missing_violations(opt_errors, context=f"DecisionEpisode:{episode_id}"))
    opts_by_id: Mapping[str, Option] = {o.option_id: o for o in opts}

    # Validate all recommendations in the episode (and collect resolved recs for review/outcome invariants)
    recs: list[Recommendation] = []
    recs_by_id: dict[str, Recommendation] = {}

    for rec_id in ep.recommendation_ids:
        try:
            rec = store.must_get(Recommendation, rec_id)
            recs.append(rec)
            recs_by_id[rec.recommendation_id] = rec
        except ResolveError as err:
            resolve_errors.append(err)
            violations.append(
                InvariantViolation(rule="missing_reference", message=f"Recommendation missing: {rec_id}")
            )
            continue

        rec_report = validate_recommendation(store, rec_id)
        violations.extend(rec_report.violations)
        resolve_errors.extend(rec_report.resolve_errors)

    # ---------------------------
    # v0.5.2 Choice invariants (NEW)
    # ---------------------------

    acted = _episode_acted(ep)

    violations.extend(
        require_choice_exists_if_episode_acted(
            episode_id=ep.episode_id,
            acted=acted,
            choice_ids=tuple(getattr(ep, "choice_ids", tuple()) or tuple()),
        )
    )

    # Resolve choices (only if present) and validate coherence.
    if getattr(ep, "choice_ids", ()):
        choices, ch_errors = store.resolve_many(ChoiceRecord, ep.choice_ids)
        resolve_errors.extend(ch_errors)

        if ch_errors:
            violations.extend(_as_missing_violations(ch_errors, context=f"DecisionEpisode:{episode_id}"))
        else:
            # Choices validate against resolved recs and opts.
            violations.extend(
                validate_choices(
                    choices=tuple(choices),
                    recommendations_by_id=recs_by_id,
                    options_by_id=opts_by_id,
                )
            )

    # ---------------------------
    # v0.5.2 Outcome invariants
    # ---------------------------

    has_rec = bool(ep.recommendation_ids)

    violations.extend(
        require_outcome_exists_if_episode_acted(
            episode_id=ep.episode_id,
            acted=acted,
            has_recommendation=has_rec,
            outcome_ids=tuple(ep.outcome_ids),
        )
    )

    outcomes_by_id: dict[str, Outcome] = {}
    if ep.outcome_ids:
        outcomes, out_errors = store.resolve_many(Outcome, ep.outcome_ids)
        resolve_errors.extend(out_errors)
        if out_errors:
            violations.extend(_as_missing_violations(out_errors, context=f"DecisionEpisode:{episode_id}"))
        else:
            outcomes_by_id = {o.outcome_id: o for o in outcomes}
            violations.extend(
                validate_outcomes(
                    outcomes=tuple(outcomes),
                    recommendations_by_id=recs_by_id,
                    options_by_id=opts_by_id,
                )
            )

    # ---------------------------
    # v0.5.2 Review invariants (existing pattern)
    # ---------------------------

    violations.extend(
        require_review_exists_if_override_used(
            episode_id=ep.episode_id,
            recommendations=recs,
            review_ids=tuple(ep.review_ids),
        )
    )

    reviews_by_id: dict[str, ReviewRecord] = {}
    if ep.review_ids:
        review_items, rev_errors = store.resolve_many(ReviewRecord, ep.review_ids)
        resolve_errors.extend(rev_errors)
        if rev_errors:
            violations.extend(_as_missing_violations(rev_errors, context=f"DecisionEpisode:{episode_id}"))
        else:
            reviews_by_id = {r.review_id: r for r in review_items}

    any_override_used = any(getattr(r, "override_used", False) is True for r in recs)
    latest_review_id = ep.latest_review_id()

    if any_override_used and latest_review_id:
        # We already resolved all reviews above; fall back to store if needed.
        review = reviews_by_id.get(latest_review_id)
        if review is None:
            review_items, rev_errors = store.resolve_many(ReviewRecord, [latest_review_id])
            resolve_errors.extend(rev_errors)
            if rev_errors:
                violations.extend(_as_missing_violations(rev_errors, context=f"DecisionEpisode:{episode_id}"))
            else:
                review = review_items[0]

        if review is not None:
            violations.extend(
                require_review_audits_overrides(
                    episode_id=ep.episode_id,
                    recommendations=recs,
                    review=review,
                )
            )

    # ---------------------------
    # v0.5.2 Calibration invariants (NEW)
    # ---------------------------

    if getattr(ep, "calibration_ids", ()):
        cals, cal_errors = store.resolve_many(CalibrationNote, ep.calibration_ids)
        resolve_errors.extend(cal_errors)

        if cal_errors:
            violations.extend(_as_missing_violations(cal_errors, context=f"DecisionEpisode:{episode_id}"))
        else:
            # If outcomes/reviews weren't resolved (no ids), keep maps empty; validator will flag missing refs.
            violations.extend(
                validate_calibrations(
                    calibrations=tuple(cals),
                    episode=ep,
                    reviews_by_id=reviews_by_id,
                    outcomes_by_id=outcomes_by_id,
                )
            )

    return ValidationReport(
        subject=f"DecisionEpisode:{episode_id}",
        violations=tuple(violations),
        resolve_errors=tuple(resolve_errors),
    )
