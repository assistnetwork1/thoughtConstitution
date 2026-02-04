from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from constitution_engine.models.option import Option
from constitution_engine.models.observation import Observation
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.runtime.store import ArtifactStore, ResolveError

from .rules import (
    InvariantViolation,
    require_observations_are_observational,
    require_recommendation_has_orientation,
    require_recommendation_has_ranked_options,
    require_proportionate_action,
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


def _as_missing_violations(errors: Sequence[ResolveError]) -> Sequence[InvariantViolation]:
    return tuple(
        InvariantViolation(
            rule="missing_reference",
            message=f"{e.artifact_type} missing: {e.artifact_id}",
        )
        for e in errors
    )


def validate_recommendation(
    store: ArtifactStore,
    rec_id: str,
) -> ValidationReport:
    rec = store.must_get(Recommendation, rec_id)

    violations: list[InvariantViolation] = []
    resolve_errors: list[ResolveError] = []

    # Basic invariants
    violations.extend(require_recommendation_has_orientation(rec))
    violations.extend(require_recommendation_has_ranked_options(rec))

    # Resolve referenced Options for proportionate action
    opt_ids = [ro.option_id for ro in rec.ranked_options]
    options, opt_errors = store.resolve_many(Option, opt_ids)
    resolve_errors.extend(opt_errors)

    options_by_id = {o.option_id: o for o in options}
    if not opt_errors:
        violations.extend(require_proportionate_action(rec, options_by_id))
    else:
        # missing options are also violations (in addition to resolve_errors)
        violations.extend(_as_missing_violations(opt_errors))

    return ValidationReport(
        subject=f"Recommendation:{rec_id}",
        violations=tuple(violations),
        resolve_errors=tuple(resolve_errors),
    )


def validate_episode(
    store: ArtifactStore,
    episode_id: str,
) -> ValidationReport:
    ep = store.must_get(DecisionEpisode, episode_id)

    violations: list[InvariantViolation] = []
    resolve_errors: list[ResolveError] = []

    # Resolve observations and validate observational purity
    observations, obs_errors = store.resolve_many(Observation, ep.observation_ids)
    resolve_errors.extend(obs_errors)
    if not obs_errors:
        violations.extend(require_observations_are_observational(observations))
    else:
        violations.extend(_as_missing_violations(obs_errors))

    # Validate all recommendations in the episode
    for rec_id in ep.recommendation_ids:
        try:
            rec_report = validate_recommendation(store, rec_id)
            violations.extend(rec_report.violations)
            resolve_errors.extend(rec_report.resolve_errors)
        except KeyError:
            err = ResolveError(artifact_type="Recommendation", artifact_id=rec_id)
            resolve_errors.append(err)
            violations.append(
                InvariantViolation(rule="missing_reference", message=f"Recommendation missing: {rec_id}")
            )

    return ValidationReport(
        subject=f"DecisionEpisode:{episode_id}",
        violations=tuple(violations),
        resolve_errors=tuple(resolve_errors),
    )
