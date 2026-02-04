from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .types import Confidence, Uncertainty, new_id, now_utc


@dataclass(frozen=True)
class Outcome:
    """
    What actually happened after an action/decision.

    Outcomes are observational-ish but anchored to a decision artifact.
    """
    outcome_id: str = field(default_factory=lambda: new_id("out"))
    created_at: datetime = field(default_factory=now_utc)

    recommendation_id: Optional[str] = None
    chosen_option_id: Optional[str] = None

    description: str = ""
    data: Any = None

    evidence_ids: Sequence[str] = field(default_factory=tuple)
    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    meta: Mapping[str, Any] = field(default_factory=dict)
