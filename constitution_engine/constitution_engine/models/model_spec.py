from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .types import new_id, now_utc


@dataclass(frozen=True)
class ModelSpec:
    """
    Declares a model family used by the engine, without hardcoding domain.
    Examples: "rule_based", "bayesian", "llm_summarizer", "scoring_v1"
    """
    model_spec_id: str = field(default_factory=lambda: new_id("ms"))
    created_at: datetime = field(default_factory=now_utc)

    name: str = ""
    family: str = ""
    version: str = "0.0.0"

    description: Optional[str] = None
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelState:
    """
    The current instantiated state of a ModelSpec (weights, priors, etc.)
    """
    model_state_id: str = field(default_factory=lambda: new_id("mst"))
    created_at: datetime = field(default_factory=now_utc)

    model_spec_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    # What influenced the state (for audit)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    observation_ids: Sequence[str] = field(default_factory=tuple)
    interpretation_ids: Sequence[str] = field(default_factory=tuple)

    notes: Optional[str] = None
