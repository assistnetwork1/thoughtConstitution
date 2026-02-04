from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional, Sequence

from .types import Confidence, Uncertainty, new_id, now_utc


@dataclass(frozen=True)
class AssumptionUpdate:
    """
    A learning artifact: how an assumption should change given outcomes.
    References assumptions by stable ID to preserve audit through renames.
    """
    assumption_id: str
    change: str  # e.g., "increase confidence", "split", "retire"
    rationale: str

    new_confidence: Optional[Confidence] = None
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReviewRecord:
    """
    Review & Compression: feed outcomes back into assumptions, weights, next questions.
    """
    review_id: str = field(default_factory=lambda: new_id("rev"))
    created_at: datetime = field(default_factory=now_utc)

    recommendation_id: Optional[str] = None
    outcome_id: Optional[str] = None

    what_happened: str = ""
    what_was_expected: str = ""
    delta: str = ""

    assumption_updates: Sequence[AssumptionUpdate] = field(default_factory=tuple)
    next_questions: Sequence[str] = field(default_factory=tuple)

    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    meta: Mapping[str, object] = field(default_factory=dict)
