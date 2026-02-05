from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from constitution_engine.models.evidence import Evidence
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.review import ReviewRecord
from constitution_engine.runtime.store import ArtifactStore, ResolveError

from .rules import (
    InvariantViolation,
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
    except KeyError:
        err = ResolveError(artifact_type="Recommendation", artifact_id=rec_id)
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
        # missing options are also violations (in addition to resolve_errors)
        violations.extend(_as_missing_violations(opt_errors, context=f"Recommendation:{rec_id}"))
    else:
        # canonical v0.5.1 gate (and optional legacy if caller toggles it in rules)
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
    except KeyError:
        err = ResolveError(artifact_type="DecisionEpisode", artifact_id=episode_id)
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

    # Validate all recommendations in the episode (and collect resolved recs for review invariants)
    recs: list[Recommendation] = []
    for rec_id in ep.recommendation_ids:
        try:
            rec = store.must_get(Recommendation, rec_id)
            recs.append(rec)
        except KeyError:
            err = ResolveError(artifact_type="Recommendation", artifact_id=rec_id)
            resolve_errors.append(err)
            violations.append(
                InvariantViolation(rule="missing_reference", message=f"Recommendation missing: {rec_id}")
            )
            continue

        rec_report = validate_recommendation(store, rec_id)
        violations.extend(rec_report.violations)
        resolve_errors.extend(rec_report.resolve_errors)

    # ---------------------------
    # v0.5.2 Review invariants
    # ---------------------------

    # INV-REV-001: overrides imply review_ids must exist
    violations.extend(
        require_review_exists_if_override_used(
            episode_id=ep.episode_id,
            recommendations=recs,
            review_ids=tuple(ep.review_ids),
        )
    )

    any_override_used = any(getattr(r, "override_used", False) is True for r in recs)
    latest_review_id = ep.latest_review_id()

    # INV-REV-002: if overrides used AND we have a review id, audit must cover overrides
    if any_override_used and latest_review_id:
        review_items, rev_errors = store.resolve_many(ReviewRecord, [latest_review_id])
        resolve_errors.extend(rev_errors)

        if rev_errors:
            violations.extend(_as_missing_violations(rev_errors, context=f"DecisionEpisode:{episode_id}"))
        else:
            review = review_items[0]
            violations.extend(
                require_review_audits_overrides(
                    episode_id=ep.episode_id,
                    recommendations=recs,
                    review=review,
                )
            )

    return ValidationReport(
        subject=f"DecisionEpisode:{episode_id}",
        violations=tuple(violations),
        resolve_errors=tuple(resolve_errors),
    )
