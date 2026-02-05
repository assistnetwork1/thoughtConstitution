# constitution_engine/models/choice.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .types import new_id, now_utc


class ChoiceBy(str, Enum):
    HUMAN = "human"
    POLICY = "policy"
    MODULE = "module"


@dataclass(frozen=True)
class ChoiceRecord:
    """
    Canonical commitment record:
    This is the bridge from "we recommended options" to "we chose option X".

    Append-only, auditable, episode-bounded.
    """
    episode_id: str
    recommendation_id: str
    option_id: str

    chosen_by: ChoiceBy = ChoiceBy.HUMAN
    rationale: str = ""
    used_override: bool = False

    choice_id: str = field(default_factory=lambda: new_id("choice"))
    created_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        # Defensive contract checks (keep these strict; they prevent silent junk).
        if not self.episode_id:
            raise ValueError("ChoiceRecord.episode_id must be non-empty")
        if not self.recommendation_id:
            raise ValueError("ChoiceRecord.recommendation_id must be non-empty")
        if not self.option_id:
            raise ValueError("ChoiceRecord.option_id must be non-empty")

        # Normalize None-like rationale to empty string (while keeping dataclass frozen).
        if self.rationale is None:
            object.__setattr__(self, "rationale", "")
