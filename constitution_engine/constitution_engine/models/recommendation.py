from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple

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

        # Normalize sequences defensively (immutability)
        object.__setattr__(self, "uncertainties", tuple(self.uncertainties))
        object.__setattr__(self, "tradeoffs", tuple(self.tradeoffs))
        object.__setattr__(self, "constraint_checks", tuple(self.constraint_checks))

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

    v0.5.1 additions (for override governance + review enforcement):
      - override_used: whether any constitutional override was required/used
      - override_scope_used: permissions actually invoked (subset of Orientation.override_scope)

    (Full lifecycle enforcement is handled by invariants + ReviewRecord; model stores the facts.)
    """
    recommendation_id: str = field(default_factory=lambda: new_id("rec"))
    created_at: datetime = field(default_factory=now_utc)

    orientation_id: str = ""
    ranked_options: Sequence[RankedOption] = field(default_factory=tuple)

    evidence_ids: Sequence[str] = field(default_factory=tuple)
    observation_ids: Sequence[str] = field(default_factory=tuple)
    interpretation_ids: Sequence[str] = field(default_factory=tuple)
    model_state_ids: Sequence[str] = field(default_factory=tuple)

    # v0.5.1 — override usage logging
    override_used: bool = False
    override_scope_used: Sequence[str] = field(default_factory=tuple)

    # v0.5.1 — justification outputs (kept permissive at model level; invariants may require non-empty)
    uncertainty_summary: Optional[str] = None
    proportionate_action_justification: Optional[str] = None

    summary: Optional[str] = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep this strict; Recommendations without Orientation violate the kernel loop.
        # (If you want to allow drafts, add an explicit draft flag rather than empty orientation_id.)
        if self.orientation_id == "":
            raise ValueError("Recommendation.orientation_id must be set (non-empty)")

        # Normalize sequences defensively to tuples so immutability is real.
        object.__setattr__(self, "ranked_options", tuple(self.ranked_options))
        _validate_contiguous_ranks(self.ranked_options)

        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "interpretation_ids", tuple(self.interpretation_ids))
        object.__setattr__(self, "model_state_ids", tuple(self.model_state_ids))

        object.__setattr__(self, "override_scope_used", tuple(self.override_scope_used))

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_summary(self, summary: Optional[str]) -> "Recommendation":
        return replace(self, summary=summary)

    def with_meta(self, **meta: object) -> "Recommendation":
        return replace(self, meta=_merge_meta(self.meta, meta))

    def with_override_used(self, used: bool) -> "Recommendation":
        return replace(self, override_used=bool(used))

    def with_override_scope_used(self, *scope_items: str) -> "Recommendation":
        return replace(self, override_scope_used=_append_unique_str(self.override_scope_used, *scope_items))

    def with_uncertainty_summary(self, text: Optional[str]) -> "Recommendation":
        return replace(self, uncertainty_summary=text)

    def with_proportionate_action_justification(self, text: Optional[str]) -> "Recommendation":
        return replace(self, proportionate_action_justification=text)

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

    def as_dict(self) -> Mapping[str, Any]:
        """
        Debug/serialization-friendly view.
        """
        return {
            "recommendation_id": self.recommendation_id,
            "created_at": self.created_at.isoformat(),
            "orientation_id": self.orientation_id,
            "ranked_options": [
                {
                    "option_id": ro.option_id,
                    "rank": ro.rank,
                    "score": ro.score,
                    "rationale": ro.rationale,
                    "confidence": getattr(ro.confidence, "value", ro.confidence),
                    "uncertainties": [
                        {
                            "description": getattr(u, "description", None),
                            "level": getattr(u, "level", None),
                        }
                        for u in ro.uncertainties
                    ],
                    "tradeoffs": list(ro.tradeoffs),
                    "constraint_checks": list(ro.constraint_checks),
                }
                for ro in self.ranked_options
            ],
            "evidence_ids": list(self.evidence_ids),
            "observation_ids": list(self.observation_ids),
            "interpretation_ids": list(self.interpretation_ids),
            "model_state_ids": list(self.model_state_ids),
            "override_used": self.override_used,
            "override_scope_used": list(self.override_scope_used),
            "uncertainty_summary": self.uncertainty_summary,
            "proportionate_action_justification": self.proportionate_action_justification,
            "summary": self.summary,
            "meta": dict(self.meta),
        }
