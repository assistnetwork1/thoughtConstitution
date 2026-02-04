from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Mapping, Optional, Sequence, Tuple

from .types import new_id, now_utc


def _as_tuple(seq: Sequence[str]) -> Tuple[str, ...]:
    return tuple(seq)


def _append_unique(seq: Sequence[str], *items: str) -> Tuple[str, ...]:
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
class Lineage:
    """
    Trace chain pointers. All fields are IDs, not embedded objects.
    """
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

    # -----------------------
    # Immutability helpers
    # -----------------------

    def add_raw_inputs(self, *ids: str) -> "Lineage":
        return replace(self, raw_input_ids=_append_unique(self.raw_input_ids, *ids))

    def add_evidence(self, *ids: str) -> "Lineage":
        return replace(self, evidence_ids=_append_unique(self.evidence_ids, *ids))

    def add_observations(self, *ids: str) -> "Lineage":
        return replace(self, observation_ids=_append_unique(self.observation_ids, *ids))

    def add_interpretations(self, *ids: str) -> "Lineage":
        return replace(self, interpretation_ids=_append_unique(self.interpretation_ids, *ids))

    def add_model_specs(self, *ids: str) -> "Lineage":
        return replace(self, model_spec_ids=_append_unique(self.model_spec_ids, *ids))

    def add_model_states(self, *ids: str) -> "Lineage":
        return replace(self, model_state_ids=_append_unique(self.model_state_ids, *ids))

    def add_orientations(self, *ids: str) -> "Lineage":
        return replace(self, orientation_ids=_append_unique(self.orientation_ids, *ids))

    def add_options(self, *ids: str) -> "Lineage":
        return replace(self, option_ids=_append_unique(self.option_ids, *ids))

    def add_recommendations(self, *ids: str) -> "Lineage":
        return replace(self, recommendation_ids=_append_unique(self.recommendation_ids, *ids))

    def add_outcomes(self, *ids: str) -> "Lineage":
        return replace(self, outcome_ids=_append_unique(self.outcome_ids, *ids))

    def add_reviews(self, *ids: str) -> "Lineage":
        return replace(self, review_ids=_append_unique(self.review_ids, *ids))

    # -----------------------
    # Convenience
    # -----------------------

    def is_empty(self) -> bool:
        return not any(
            bool(_as_tuple(getattr(self, f)))
            for f in (
                "raw_input_ids",
                "evidence_ids",
                "observation_ids",
                "interpretation_ids",
                "model_spec_ids",
                "model_state_ids",
                "orientation_ids",
                "option_ids",
                "recommendation_ids",
                "outcome_ids",
                "review_ids",
            )
        )


@dataclass(frozen=True)
class AuditTrail:
    """
    A top-level auditable record for a decision episode or artifact.

    You can attach this to any persisted artifact in storage.
    """
    audit_id: str = field(default_factory=lambda: new_id("aud"))
    created_at: datetime = field(default_factory=now_utc)

    subject_id: str = ""          # the artifact being audited (e.g., recommendation_id)
    subject_type: str = ""        # freeform: "Recommendation", "Outcome", etc.

    lineage: Lineage = field(default_factory=Lineage)

    # Optional: immutable digest summary, or storage pointers
    notes: Optional[str] = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.subject_id == "":
            raise ValueError("AuditTrail.subject_id must be set (non-empty)")
        if self.subject_type == "":
            raise ValueError("AuditTrail.subject_type must be set (non-empty)")

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_notes(self, notes: Optional[str]) -> "AuditTrail":
        return replace(self, notes=notes)

    def with_meta(self, **meta: object) -> "AuditTrail":
        return replace(self, meta=_merge_meta(self.meta, meta))

    def with_lineage(self, lineage: Lineage) -> "AuditTrail":
        return replace(self, lineage=lineage)

    # Convenience: mutate lineage immutably in one call
    def add_lineage(
        self,
        *,
        raw_input_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        observation_ids: Sequence[str] = (),
        interpretation_ids: Sequence[str] = (),
        model_spec_ids: Sequence[str] = (),
        model_state_ids: Sequence[str] = (),
        orientation_ids: Sequence[str] = (),
        option_ids: Sequence[str] = (),
        recommendation_ids: Sequence[str] = (),
        outcome_ids: Sequence[str] = (),
        review_ids: Sequence[str] = (),
    ) -> "AuditTrail":
        lin = self.lineage
        if raw_input_ids:
            lin = lin.add_raw_inputs(*raw_input_ids)
        if evidence_ids:
            lin = lin.add_evidence(*evidence_ids)
        if observation_ids:
            lin = lin.add_observations(*observation_ids)
        if interpretation_ids:
            lin = lin.add_interpretations(*interpretation_ids)
        if model_spec_ids:
            lin = lin.add_model_specs(*model_spec_ids)
        if model_state_ids:
            lin = lin.add_model_states(*model_state_ids)
        if orientation_ids:
            lin = lin.add_orientations(*orientation_ids)
        if option_ids:
            lin = lin.add_options(*option_ids)
        if recommendation_ids:
            lin = lin.add_recommendations(*recommendation_ids)
        if outcome_ids:
            lin = lin.add_outcomes(*outcome_ids)
        if review_ids:
            lin = lin.add_reviews(*review_ids)
        return replace(self, lineage=lin)
