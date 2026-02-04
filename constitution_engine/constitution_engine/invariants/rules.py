from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from constitution_engine.models.types import InfoType
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import Recommendation


@dataclass(frozen=True)
class InvariantViolation:
    rule: str
    message: str


def require_observations_are_observational(
    observations: Iterable[Observation],
) -> Sequence[InvariantViolation]:
    violations = []
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


def require_recommendation_has_orientation(rec: Recommendation) -> Sequence[InvariantViolation]:
    if not rec.orientation_id:
        return (InvariantViolation(rule="recommendation_orientation", message="Recommendation missing orientation_id"),)
    return tuple()


def require_recommendation_has_ranked_options(rec: Recommendation) -> Sequence[InvariantViolation]:
    if not rec.ranked_options:
        return (InvariantViolation(rule="recommendation_ranked_options", message="Recommendation has no ranked_options"),)
    return tuple()


def require_proportionate_action(
    rec: Recommendation,
    options_by_id: Mapping[str, Option],
    *,
    high_uncertainty: float = 0.7,
    low_reversibility: float = 0.3,
    nontrivial_impact: float = 0.4,
) -> Sequence[InvariantViolation]:
    """
    Proportionate Action invariant:

    If the recommendation includes at least one EXECUTE option that is:
      - high uncertainty (max uncertainty >= threshold), and
      - low reversibility, and
      - non-trivial impact,
    then the recommendation must also include at least one:
      - INFO_GATHERING or HEDGE option.

    This forces "hedge / learn first" behavior when stakes are high and undo is hard.
    """
    violations = []

    ranked_option_ids = [ro.option_id for ro in rec.ranked_options]
    ranked_options = []
    missing = []
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

    has_risky_execute = any(
        (opt.kind == OptionKind.EXECUTE)
        and (max_unc(opt) >= high_uncertainty)
        and (opt.reversibility.value <= low_reversibility)
        and (opt.impact.value >= nontrivial_impact)
        for opt in ranked_options
    )

    if not has_risky_execute:
        return tuple()

    has_hedge_or_learn = any(
        opt.kind in {OptionKind.HEDGE, OptionKind.INFO_GATHERING}
        for opt in ranked_options
    )

    if not has_hedge_or_learn:
        violations.append(
            InvariantViolation(
                rule="proportionate_action",
                message=(
                    "High-uncertainty, low-reversibility, non-trivial impact EXECUTE option present, "
                    "but no HEDGE or INFO_GATHERING option included in the Recommendation."
                ),
            )
        )

    return tuple(violations)
