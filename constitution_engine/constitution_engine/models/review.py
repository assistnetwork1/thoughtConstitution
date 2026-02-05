from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional

from .types import new_id, now_utc


def _merge_meta(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Mapping[str, Any]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


@dataclass(frozen=True)
class ReviewRecord:
    """
    Episode-scoped review artifact.

    Purpose (v0.5.2 thin-slice):
      - record outcomes + calibration notes
      - provide override audit visibility when constitutional overrides were used
    """
    review_id: str = field(default_factory=lambda: new_id("rev"))
    created_at: datetime = field(default_factory=now_utc)

    episode_id: Optional[str] = None

    outcome_summary: str = ""
    calibration_notes: str = ""

    # Minimal override audit container.
    # Expected shape (thin-slice):
    # {
    #   "overrides": [
    #       {
    #           "recommendation_id": "...",
    #           "override_scope_used": ["PERM_A", ...],
    #           "rationale": "..."
    #       },
    #       ...
    #   ]
    # }
    override_audit: Mapping[str, Any] = field(default_factory=dict)

    meta: Mapping[str, Any] = field(default_factory=dict)

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_episode(self, episode_id: Optional[str]) -> "ReviewRecord":
        return replace(self, episode_id=episode_id)

    def with_outcome_summary(self, outcome_summary: str) -> "ReviewRecord":
        return replace(self, outcome_summary=outcome_summary)

    def with_calibration_notes(self, calibration_notes: str) -> "ReviewRecord":
        return replace(self, calibration_notes=calibration_notes)

    def with_override_audit(self, override_audit: Mapping[str, Any]) -> "ReviewRecord":
        return replace(self, override_audit=dict(override_audit))

    def with_meta(self, **meta: Any) -> "ReviewRecord":
        return replace(self, meta=_merge_meta(self.meta, meta))
        
@dataclass(frozen=True)
class AssumptionUpdate:
    """
    Minimal v0.5.x placeholder to keep the public import contract stable.

    Represents a reviewed/explicit change to an assumption (or assumption-like artifact).
    This is NOT Bayesian updating; it is auditable revision logged via Review.
    """
    update_id: str = field(default_factory=lambda: new_id("asmpu"))
    created_at: datetime = field(default_factory=now_utc)

    # What was changed (IDs only, audit-first)
    assumption_id: Optional[str] = None
    prior_artifact_id: Optional[str] = None
    new_artifact_id: Optional[str] = None

    # Human explanation / audit trace
    rationale: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)