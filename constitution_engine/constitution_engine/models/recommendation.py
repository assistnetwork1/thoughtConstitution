from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Mapping, Optional, Sequence, Tuple, Any

from .types import Confidence, Uncertainty, new_id, now_utc


def _as_tuple_str(seq: Sequence[str]) -> Tuple[str, ...]:
    return tuple(seq)


def _as_tuple_ro(seq: Sequence["RankedOption"]) -> Tuple["RankedOption", ...]:
    return tuple(seq)


def _as_tuple_unc(seq: Sequence[Uncertainty]) -> Tuple[Uncertainty, ...]:
    return tuple(seq)


def _append_unique_str(seq: Sequence[str], *items: str) -> Tuple[str, ...]:
    out = list(_as_tuple_str(seq))
    seen = set(out)
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return tuple(out)


def _append_uncertainty(seq: Sequence[Uncertainty], *items: Uncertainty) -> Tuple[Uncertainty, ...]:
    out = list(_as_tuple_unc(seq))
    out.extend([u for u in items if u is not None])
    return tuple(out)


def _merge_meta(base: Mapping[str, object], patch: Mapping[str, object]) -> Mapping[str, object]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


def _validate_contiguous_ranks(ranked_options: Sequence["RankedOption"]) -> None:
    ranks = [ro.rank for ro in ranked_options]
    if not ranks:
        return
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("RankedOption.rank must be contiguous starting at 1")
    if len(set(ranks)) != len(ranks):
        raise ValueError("RankedOption.rank values must be unique")


@dataclass(frozen=True)
class RankedOption:
    """
    An option with ranking metadata and explanation.
    """
    option_id: str
    rank: int

    # score is a ranking signal, not necessarily a normative push
    score: float
    rationale: str

    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    tradeoffs: Sequence[str] = field(default_factory=tuple)
    constraint_checks: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("RankedOption.rank must be >= 1")
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError("RankedOption.score must be between 0.0 and 1.0")

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_rationale(self, rationale: str) -> "RankedOption":
        return replace(self, rationale=rationale)

    def with_score(self, score: float) -> "RankedOption":
        if not (0.0 <= float(score) <= 1.0):
            raise ValueError("RankedOption.score must be between 0.0 and 1.0")
        return replace(self, score=float(score))

    def with_confidence(self, confidence: Confidence) -> "RankedOption":
        return replace(self, confidence=confidence)

    def add_uncertainties(self, *uncertainties: Uncertainty) -> "RankedOption":
        return replace(self, uncertainties=_append_uncertainty(self.uncertainties, *uncertainties))

    def add_tradeoffs(self, *tradeoffs: str) -> "RankedOption":
        return replace(self, tradeoffs=_append_unique_str(self.tradeoffs, *tradeoffs))

    def add_constraint_checks(self, *checks: str) -> "RankedOption":
        return replace(self, constraint_checks=_append_unique_str(self.constraint_checks, *checks))


@dataclass(frozen=True)
class Recommendation:
    """
    Ranked, explainable, uncertainty-aware action proposals.
    """
    recommendation_id: str = field(default_factory=lambda: new_id("rec"))
    created_at: datetime = field(default_factory=now_utc)

    orientation_id: str = ""
    ranked_options: Sequence[RankedOption] = field(default_factory=tuple)

    evidence_ids: Sequence[str] = field(default_factory=tuple)
    observation_ids: Sequence[str] = field(default_factory=tuple)
    interpretation_ids: Sequence[str] = field(default_factory=tuple)
    model_state_ids: Sequence[str] = field(default_factory=tuple)

    summary: Optional[str] = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_contiguous_ranks(self.ranked_options)
        if self.orientation_id == "":
            # Keep this strict; Recommendations without Orientation violate the kernel loop.
            # (If you want to allow drafts, add an explicit draft flag rather than empty orientation_id.)
            raise ValueError("Recommendation.orientation_id must be set (non-empty)")

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_summary(self, summary: Optional[str]) -> "Recommendation":
        return replace(self, summary=summary)

    def with_meta(self, **meta: object) -> "Recommendation":
        return replace(self, meta=_merge_meta(self.meta, meta))

    def add_ranked_options(self, *ranked_options: RankedOption) -> "Recommendation":
        new_ros = _as_tuple_ro(self.ranked_options) + tuple(ranked_options)
        # Ensure ranks remain contiguous; easiest is to require caller to supply rank numbers properly.
        _validate_contiguous_ranks(new_ros)
        return replace(self, ranked_options=new_ros)

    def add_evidence(self, *evidence_ids: str) -> "Recommendation":
        return replace(self, evidence_ids=_append_unique_str(self.evidence_ids, *evidence_ids))

    def add_observations(self, *observation_ids: str) -> "Recommendation":
        return replace(self, observation_ids=_append_unique_str(self.observation_ids, *observation_ids))

    def add_interpretations(self, *interpretation_ids: str) -> "Recommendation":
        return replace(self, interpretation_ids=_append_unique_str(self.interpretation_ids, *interpretation_ids))

    def add_model_states(self, *model_state_ids: str) -> "Recommendation":
        return replace(self, model_state_ids=_append_unique_str(self.model_state_ids, *model_state_ids))

    # -----------------------
    # Convenience
    # -----------------------

    def top_option_id(self) -> Optional[str]:
        if not self.ranked_options:
            return None
        # ranks are contiguous; rank=1 is top
        top = min(self.ranked_options, key=lambda ro: ro.rank)
        return top.option_id

    def ids_for_trace(self) -> Mapping[str, Sequence[str]]:
        """
        A compact, consistent way to expose provenance pointers for audit trail building.
        """
        return {
            "evidence_ids": tuple(self.evidence_ids),
            "observation_ids": tuple(self.observation_ids),
            "interpretation_ids": tuple(self.interpretation_ids),
            "model_state_ids": tuple(self.model_state_ids),
        }
