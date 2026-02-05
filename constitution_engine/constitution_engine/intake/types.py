from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    statement: str
    horizon_days: int
    success_criteria: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RawInputItem:
    raw_id: str
    text: str
    source_uri: str  # e.g. "internal:user", "url:https://...", "doc:..."
    created_at_utc: str


@dataclass(frozen=True)
class MissingInput:
    field: str
    question: str
    severity: str  # "LOW" | "MED" | "HIGH"


@dataclass(frozen=True)
class AdapterPolicy:
    default_confidence: float = 0.6
    default_uncertainty: float = 0.4
    auto_probe_on_missing: bool = True
    allow_commit_proposals: bool = True


# Final output of the adapter: a "draft" bundle ready for user edits + kernel validation.
@dataclass(frozen=True)
class DraftEpisode:
    episode_id: str
    goal: GoalSpec

    evidence: Tuple["Evidence", ...]
    observations: Tuple["Observation", ...]
    interpretations: Tuple["Interpretation", ...]
    options: Tuple["Option", ...]
    recommendation: "Recommendation | None"

    missing_inputs: Tuple[MissingInput, ...]
    notes: Tuple[str, ...] = ()


# Forward refs for type checkers (models live elsewhere).
from constitution_engine.models.evidence import Evidence  # noqa: E402
from constitution_engine.models.observation import Observation  # noqa: E402
from constitution_engine.models.option import Option  # noqa: E402
from constitution_engine.models.recommendation import Recommendation  # noqa: E402
from constitution_engine.models.interpretation import Interpretation  # noqa: E402
