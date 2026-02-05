from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from .types import Confidence, Uncertainty, new_id, now_utc


@dataclass(frozen=True)
class CalibrationNote:
    """
    A human-authored, auditable learning artifact.

    Records what should change *next time* based on:
      - an Episode,
      - its Outcomes,
      - and a ReviewRecord.

    Append-only. Does not mutate prior artifacts.
    """
    calibration_id: str = field(default_factory=lambda: new_id("cal"))

    created_at: datetime = field(default_factory=now_utc)

    episode_id: str = ""
    review_id: str = ""
    outcome_ids: Sequence[str] = field(default_factory=tuple)

    summary: str = ""
    proposed_changes: Sequence[str] = field(default_factory=tuple)

    # Optional structured hints for future adapters (safe, not executable)
    patch: Mapping[str, Any] = field(default_factory=dict)

    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    meta: Mapping[str, Any] = field(default_factory=dict)
