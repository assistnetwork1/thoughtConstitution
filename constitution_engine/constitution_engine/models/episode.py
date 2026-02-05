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

    # NEW: explicit commitment records (IDs-only, append-only)
    choice_ids: Sequence[str] = field(default_factory=tuple)

    outcome_ids: Sequence[str] = field(default_factory=tuple)
    review_ids: Sequence[str] = field(default_factory=tuple)
    calibration_ids: Sequence[str] = field(default_factory=tuple)
    audit_ids: Sequence[str] = field(default_factory=tuple)

    # Act marker (makes the invariant “acted ⇒ outcome” precise)
    acted: bool = False
    acted_at: Optional[datetime] = None

    # NOTE: We intentionally keep chosen_option_id for compatibility/ergonomics.
    # It can be treated as "primary/first choice" for thin-slice flows.
    chosen_option_id: Optional[str] = None

    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize sequences defensively so immutability is real even if callers pass lists.
        object.__setattr__(self, "raw_input_ids", _as_tuple(self.raw_input_ids))
        object.__setattr__(self, "evidence_ids", _as_tuple(self.evidence_ids))
        object.__setattr__(self, "observation_ids", _as_tuple(self.observation_ids))
        object.__setattr__(self, "interpretation_ids", _as_tuple(self.interpretation_ids))
        object.__setattr__(self, "model_spec_ids", _as_tuple(self.model_spec_ids))
        object.__setattr__(self, "model_state_ids", _as_tuple(self.model_state_ids))
        object.__setattr__(self, "orientation_ids", _as_tuple(self.orientation_ids))
        object.__setattr__(self, "option_ids", _as_tuple(self.option_ids))
        object.__setattr__(self, "recommendation_ids", _as_tuple(self.recommendation_ids))
        object.__setattr__(self, "choice_ids", _as_tuple(self.choice_ids))
        object.__setattr__(self, "outcome_ids", _as_tuple(self.outcome_ids))
        object.__setattr__(self, "review_ids", _as_tuple(self.review_ids))
        object.__setattr__(self, "calibration_ids", _as_tuple(self.calibration_ids))
        object.__setattr__(self, "audit_ids", _as_tuple(self.audit_ids))

        # Minimal consistency: if acted is True, we *prefer* chosen_option_id set,
        # but we do not hard-fail here; invariants enforce strictness.
        if self.acted and not self.chosen_option_id and not self.choice_ids:
            pass

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

    def add_choices(self, *choice_ids: str) -> "DecisionEpisode":
        return replace(self, choice_ids=_append_unique(self.choice_ids, *choice_ids))

    def add_outcomes(self, *outcome_ids: str) -> "DecisionEpisode":
        return replace(self, outcome_ids=_append_unique(self.outcome_ids, *outcome_ids))

    def add_reviews(self, *review_ids: str) -> "DecisionEpisode":
        return replace(self, review_ids=_append_unique(self.review_ids, *review_ids))

    def add_calibrations(self, *calibration_ids: str) -> "DecisionEpisode":
        return replace(self, calibration_ids=_append_unique(self.calibration_ids, *calibration_ids))

    def add_audits(self, *audit_ids: str) -> "DecisionEpisode":
        return replace(self, audit_ids=_append_unique(self.audit_ids, *audit_ids))

    # -----------------------
    # Act helpers
    # -----------------------

    def mark_acted(
        self,
        *,
        chosen_option_id: Optional[str] = None,
        acted_at: Optional[datetime] = None,
    ) -> "DecisionEpisode":
        """
        Marks the episode as having taken action (picked/executed an option).

        NOTE:
        - The *strict* rule "acted ⇒ has ChoiceRecord" is enforced by invariants,
          not here, to keep the dataclass permissive for partial construction in tests.
        """
        # If caller provides a chosen option, preserve it (compat); otherwise keep existing.
        if chosen_option_id is not None and not chosen_option_id:
            raise ValueError("chosen_option_id must be non-empty when provided")

        # Defensive: if already acted with a different option, refuse the mutation.
        if (
            chosen_option_id
            and self.acted
            and self.chosen_option_id
            and self.chosen_option_id != chosen_option_id
        ):
            raise ValueError(
                f"Episode already acted on {self.chosen_option_id}; cannot change to {chosen_option_id}."
            )

        return replace(
            self,
            acted=True,
            chosen_option_id=chosen_option_id or self.chosen_option_id,
            acted_at=acted_at or (self.acted_at or now_utc()),
        )

    # Optional convenience: a semantic alias some callers prefer.
    def act_on_option(self, option_id: str, *, acted_at: Optional[datetime] = None) -> "DecisionEpisode":
        if not option_id:
            raise ValueError("option_id is required")
        return self.mark_acted(chosen_option_id=option_id, acted_at=acted_at)

    def log_choice(self, choice_id: str) -> "DecisionEpisode":
        """
        Append a ChoiceRecord reference deterministically (idempotent).
        """
        if not choice_id:
            raise ValueError("choice_id is required")
        return replace(self, choice_ids=_append_unique(self.choice_ids, choice_id))

    def log_outcome(self, outcome_id: str) -> "DecisionEpisode":
        """
        Append an outcome reference deterministically (idempotent).
        Does not force acted=True (some systems may record outcomes before formal act).
        Invariants can enforce ordering if desired.
        """
        if not outcome_id:
            raise ValueError("outcome_id is required")
        return replace(self, outcome_ids=_append_unique(self.outcome_ids, outcome_id))

    def log_review(self, review_id: str) -> "DecisionEpisode":
        """
        Append a review reference deterministically (idempotent).
        """
        if not review_id:
            raise ValueError("review_id is required")
        return replace(self, review_ids=_append_unique(self.review_ids, review_id))

    def log_calibration(self, calibration_id: str) -> "DecisionEpisode":
        """
        Append a calibration reference deterministically (idempotent).
        """
        if not calibration_id:
            raise ValueError("calibration_id is required")
        return replace(self, calibration_ids=_append_unique(self.calibration_ids, calibration_id))

    def log_audit(self, audit_id: str) -> "DecisionEpisode":
        """
        Append an audit reference deterministically (idempotent).
        """
        if not audit_id:
            raise ValueError("audit_id is required")
        return replace(self, audit_ids=_append_unique(self.audit_ids, audit_id))

    # -----------------------
    # Convenience selectors
    # -----------------------

    def latest_orientation_id(self) -> Optional[str]:
        ids = _as_tuple(self.orientation_ids)
        return ids[-1] if ids else None

    def latest_recommendation_id(self) -> Optional[str]:
        ids = _as_tuple(self.recommendation_ids)
        return ids[-1] if ids else None

    def latest_choice_id(self) -> Optional[str]:
        ids = _as_tuple(self.choice_ids)
        return ids[-1] if ids else None

    def latest_outcome_id(self) -> Optional[str]:
        ids = _as_tuple(self.outcome_ids)
        return ids[-1] if ids else None

    def latest_review_id(self) -> Optional[str]:
        ids = _as_tuple(self.review_ids)
        return ids[-1] if ids else None

    def latest_calibration_id(self) -> Optional[str]:
        ids = _as_tuple(self.calibration_ids)
        return ids[-1] if ids else None

    def latest_audit_id(self) -> Optional[str]:
        ids = _as_tuple(self.audit_ids)
        return ids[-1] if ids else None
