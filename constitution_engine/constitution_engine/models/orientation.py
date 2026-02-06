from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple

from .types import RiskPosture, Weight, new_id, now_utc


def _as_tuple(seq: Sequence) -> Tuple:
    return tuple(seq)


def _append(seq: Sequence, *items) -> Tuple:
    out = list(_as_tuple(seq))
    out.extend([it for it in items if it is not None])
    return tuple(out)


def _append_unique_by_name(seq: Sequence, *items) -> Tuple:
    """
    Deduplicate by `.name` when present; otherwise fall back to full object identity.
    This keeps orientation stable and avoids accidental duplicates.
    """
    out = list(_as_tuple(seq))
    seen_names = {getattr(x, "name", None) for x in out}
    for it in items:
        if it is None:
            continue
        nm = getattr(it, "name", None)
        if nm is None:
            out.append(it)
            continue
        if nm not in seen_names:
            out.append(it)
            seen_names.add(nm)
    return tuple(out)


def _merge_meta(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Mapping[str, Any]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


@dataclass(frozen=True)
class Objective:
    """
    A goal the system is trying to advance.
    """
    name: str
    description: str
    weight: Weight = Weight(1.0)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Objective.name must be non-empty")


@dataclass(frozen=True)
class Constraint:
    """
    A hard boundary (must-not-violate).
    """
    name: str
    description: str
    # Optional machine-readable expression (left as string for kernel simplicity)
    expression: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Constraint.name must be non-empty")


@dataclass(frozen=True)
class ValueSignal:
    """
    A normative value or preference signal (not a constraint).
    """
    name: str
    description: str
    weight: Weight = Weight(1.0)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ValueSignal.name must be non-empty")


@dataclass(frozen=True)
class Orientation:
    """
    Explicit values/objectives/constraints and risk posture.

    This is required before producing Recommendations.
    """
    orientation_id: str = field(default_factory=lambda: new_id("ori"))
    created_at: datetime = field(default_factory=now_utc)

    objectives: Tuple[Objective, ...] = field(default_factory=tuple)
    constraints: Tuple[Constraint, ...] = field(default_factory=tuple)
    values: Tuple[ValueSignal, ...] = field(default_factory=tuple)

    risk_posture: RiskPosture = RiskPosture.BALANCED

    # Optional: context about who/what set this orientation
    owner: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize sequences -> tuples for immutability + audit stability
        object.__setattr__(self, "objectives", _as_tuple(self.objectives))
        object.__setattr__(self, "constraints", _as_tuple(self.constraints))
        object.__setattr__(self, "values", _as_tuple(self.values))
        object.__setattr__(self, "meta", dict(self.meta) if self.meta else {})

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_owner(self, owner: Optional[str]) -> "Orientation":
        return replace(self, owner=owner)

    def with_risk_posture(self, risk_posture: RiskPosture) -> "Orientation":
        return replace(self, risk_posture=risk_posture)

    def with_meta(self, **meta: Any) -> "Orientation":
        return replace(self, meta=_merge_meta(self.meta, meta))

    # -----------------------
    # Adders (stable + mostly deduped)
    # -----------------------

    def add_objectives(self, *objectives: Objective) -> "Orientation":
        return replace(self, objectives=_append_unique_by_name(self.objectives, *objectives))

    def add_constraints(self, *constraints: Constraint) -> "Orientation":
        return replace(self, constraints=_append_unique_by_name(self.constraints, *constraints))

    def add_values(self, *values: ValueSignal) -> "Orientation":
        return replace(self, values=_append_unique_by_name(self.values, *values))

    # -----------------------
    # Convenience
    # -----------------------

    def objective_weight_map(self) -> Mapping[str, float]:
        """
        Returns objective weights as plain floats for downstream scoring engines.
        """
        return {o.name: float(o.weight) for o in self.objectives}

    def value_weight_map(self) -> Mapping[str, float]:
        return {v.name: float(v.weight) for v in self.values}

    def constraint_names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.constraints)
