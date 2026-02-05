from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from constitution_engine.intake.types import AdapterPolicy, GoalSpec, MissingInput, RawInputItem


@dataclass(frozen=True)
class ObservationDraft:
    statement: str
    confidence: float | None = None
    uncertainty: float | None = None
    # Keep info_type optional so adapter can default or map.
    info_type: str | None = None


@dataclass(frozen=True)
class InterpretationDraft:
    statement: str
    confidence: float | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class OptionDraft:
    name: str
    description: str
    impact: float | None = None
    reversibility: float | None = None
    uncertainties: Tuple[float, ...] = ()
    option_kind: str | None = None          # e.g. "info_gathering"
    action_class: str | None = None         # "PROBE" | "LIMITED" | "COMMIT"


@dataclass(frozen=True)
class RecommendationDraft:
    ranked_option_names: Tuple[str, ...]
    justification: str
    override_used: bool = False
    override_scope_used: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftBundle:
    observations: Tuple[ObservationDraft, ...] = ()
    interpretations: Tuple[InterpretationDraft, ...] = ()
    options: Tuple[OptionDraft, ...] = ()
    recommendation: RecommendationDraft | None = None
    missing_inputs: Tuple[MissingInput, ...] = ()


class Drafter(Protocol):
    def draft(
        self,
        *,
        goal: GoalSpec,
        raw_inputs: list[RawInputItem],
        policy: AdapterPolicy,
    ) -> DraftBundle:
        ...
