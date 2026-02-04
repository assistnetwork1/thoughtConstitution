from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Mapping, Optional, Sequence, Tuple, TypeVar

from .types import new_id, now_utc


_T = TypeVar("_T")


def _as_tuple(seq: Sequence[str]) -> Tuple[str, ...]:
    # normalize any incoming sequence to an immutable tuple
    return tuple(seq)


def _append_unique(seq: Sequence[str], *items: str) -> Tuple[str, ...]:
    """
    Append IDs while preserving order and avoiding duplicates.
    Deterministic + stable for audit/indexing.
    """
    out = list(_as_tuple(seq))
    seen = set(out)
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return tuple(out)


def _merge_meta(base: Mapping[str, object], patch: Mapping[str, object]) -> Mapping[str, object]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


@dataclass(frozen=True)
class DecisionEpisode:
    """
    A thin container that indexes a whole Observe→Model→Orient→Act→Review loop.

    This is intentionally IDs-only (no embedded objects) to keep storage/audit clean.
    """
    episode_id: str = field(default_factory=lambda: new_id("ep"))
    created_at: datetime = field(default_factory=now_utc)

    title: Optional[str] = None
    description: Optional[str] = None

    raw_input_ids: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    observation_ids: Sequence[str] = field(default_factory=tuple)
    interpretation_ids: Sequence[str] = field(default_factory=tuple)
    model_spec_ids: Sequence[str] = field(default_factory=tuple)
    model_state_ids: Sequence[str] = field(default_factory=tuple)
    orientation_ids: Sequence[str] = field(default_factory=tuple)
    option_ids: Sequence[str] = field(default_factory=tuple)
    recommendation_ids: Sequence[str] = field(default_factory=tuple)
    outcome_ids: Sequence[str] = field(default_factory=tuple)
    review_ids: Sequence[str] = field(default_factory=tuple)
    audit_ids: Sequence[str] = field(default_factory=tuple)

    meta: Mapping[str, object] = field(default_factory=dict)

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_title(self, title: Optional[str]) -> "DecisionEpisode":
        return replace(self, title=title)

    def with_description(self, description: Optional[str]) -> "DecisionEpisode":
        return replace(self, description=description)

    def with_meta(self, **meta: object) -> "DecisionEpisode":
        return replace(self, meta=_merge_meta(self.meta, meta))

    # -----------------------
    # Adders (dedupe + stable)
    # -----------------------

    def add_raw_inputs(self, *raw_input_ids: str) -> "DecisionEpisode":
        return replace(self, raw_input_ids=_append_unique(self.raw_input_ids, *raw_input_ids))

    def add_evidence(self, *evidence_ids: str) -> "DecisionEpisode":
        return replace(self, evidence_ids=_append_unique(self.evidence_ids, *evidence_ids))

    def add_observations(self, *observation_ids: str) -> "DecisionEpisode":
        return replace(self, observation_ids=_append_unique(self.observation_ids, *observation_ids))

    def add_interpretations(self, *interpretation_ids: str) -> "DecisionEpisode":
        return replace(self, interpretation_ids=_append_unique(self.interpretation_ids, *interpretation_ids))

    def add_model_specs(self, *model_spec_ids: str) -> "DecisionEpisode":
        return replace(self, model_spec_ids=_append_unique(self.model_spec_ids, *model_spec_ids))

    def add_model_states(self, *model_state_ids: str) -> "DecisionEpisode":
        return replace(self, model_state_ids=_append_unique(self.model_state_ids, *model_state_ids))

    def add_orientations(self, *orientation_ids: str) -> "DecisionEpisode":
        return replace(self, orientation_ids=_append_unique(self.orientation_ids, *orientation_ids))

    def add_options(self, *option_ids: str) -> "DecisionEpisode":
        return replace(self, option_ids=_append_unique(self.option_ids, *option_ids))

    def add_recommendations(self, *recommendation_ids: str) -> "DecisionEpisode":
        return replace(self, recommendation_ids=_append_unique(self.recommendation_ids, *recommendation_ids))

    def add_outcomes(self, *outcome_ids: str) -> "DecisionEpisode":
        return replace(self, outcome_ids=_append_unique(self.outcome_ids, *outcome_ids))

    def add_reviews(self, *review_ids: str) -> "DecisionEpisode":
        return replace(self, review_ids=_append_unique(self.review_ids, *review_ids))

    def add_audits(self, *audit_ids: str) -> "DecisionEpisode":
        return replace(self, audit_ids=_append_unique(self.audit_ids, *audit_ids))

    # -----------------------
    # Convenience selectors
    # -----------------------

    def latest_orientation_id(self) -> Optional[str]:
        ids = _as_tuple(self.orientation_ids)
        return ids[-1] if ids else None

    def latest_recommendation_id(self) -> Optional[str]:
        ids = _as_tuple(self.recommendation_ids)
        return ids[-1] if ids else None

    def latest_outcome_id(self) -> Optional[str]:
        ids = _as_tuple(self.outcome_ids)
        return ids[-1] if ids else None

    def latest_review_id(self) -> Optional[str]:
        ids = _as_tuple(self.review_ids)
        return ids[-1] if ids else None

    def latest_audit_id(self) -> Optional[str]:
        ids = _as_tuple(self.audit_ids)
        return ids[-1] if ids else None
