from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Sequence, Tuple, Optional

from .types import InfoType, Confidence, Uncertainty, new_id, now_utc


_INTERPRETIVE_TYPES = {
    InfoType.CLAIM,
    InfoType.EXPLANATION,
    InfoType.HYPOTHESIS,
    InfoType.FRAME,
}


def _as_tuple_str(seq: Sequence[str]) -> Tuple[str, ...]:
    return tuple(seq)


def _as_tuple_unc(seq: Sequence[Uncertainty]) -> Tuple[Uncertainty, ...]:
    return tuple(seq)


def _as_tuple_asm(seq: Sequence["Assumption"]) -> Tuple["Assumption", ...]:
    return tuple(seq)


def _append_unique_str(seq: Sequence[str], *items: str) -> Tuple[str, ...]:
    out = list(_as_tuple_str(seq))
    seen = set(out)
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return tuple(out)


def _append_uncertainty(seq: Sequence[Uncertainty], *items: Uncertainty) -> Tuple[Uncertainty, ...]:
    out = list(_as_tuple_unc(seq))
    out.extend([u for u in items if u is not None])
    return tuple(out)


def _append_assumptions(seq: Sequence["Assumption"], *items: "Assumption") -> Tuple["Assumption", ...]:
    """
    Append assumptions while deduping by assumption_id (stable identity).
    """
    out = list(_as_tuple_asm(seq))
    seen = {a.assumption_id for a in out}
    for a in items:
        if a is None:
            continue
        if a.assumption_id not in seen:
            out.append(a)
            seen.add(a.assumption_id)
    return tuple(out)


def _merge_meta(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Mapping[str, Any]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


@dataclass(frozen=True)
class Assumption:
    """
    A named assumption with explicit uncertainty.
    Uses stable IDs to keep audit robust through renames.
    """
    assumption_id: str = field(default_factory=lambda: new_id("asm"))
    name: str = ""
    description: str = ""

    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.name.strip() == "" and self.description.strip() == "":
            # allow stubs, but prefer having at least one
            pass

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_name(self, name: str) -> "Assumption":
        return replace(self, name=name)

    def with_description(self, description: str) -> "Assumption":
        return replace(self, description=description)

    def with_confidence(self, confidence: Confidence) -> "Assumption":
        return replace(self, confidence=confidence)

    def add_uncertainties(self, *uncertainties: Uncertainty) -> "Assumption":
        return replace(self, uncertainties=_append_uncertainty(self.uncertainties, *uncertainties))

    def max_uncertainty_level(self) -> Optional[float]:
        if not self.uncertainties:
            return None
        return max(u.level for u in self.uncertainties)


@dataclass(frozen=True)
class Interpretation:
    """
    Structured hypothesis / explanation that connects observations.
    """
    interpretation_id: str = field(default_factory=lambda: new_id("int"))
    created_at: datetime = field(default_factory=now_utc)

    info_type: InfoType = InfoType.HYPOTHESIS
    title: str = ""
    narrative: str = ""

    observation_ids: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)

    assumptions: Sequence[Assumption] = field(default_factory=tuple)
    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    model_payload: Any = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.info_type not in _INTERPRETIVE_TYPES:
            raise ValueError("Interpretation.info_type must be interpretive")

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_title(self, title: str) -> "Interpretation":
        return replace(self, title=title)

    def with_narrative(self, narrative: str) -> "Interpretation":
        return replace(self, narrative=narrative)

    def with_confidence(self, confidence: Confidence) -> "Interpretation":
        return replace(self, confidence=confidence)

    def with_model_payload(self, payload: Any) -> "Interpretation":
        return replace(self, model_payload=payload)

    def add_observations(self, *observation_ids: str) -> "Interpretation":
        return replace(self, observation_ids=_append_unique_str(self.observation_ids, *observation_ids))

    def add_evidence(self, *evidence_ids: str) -> "Interpretation":
        return replace(self, evidence_ids=_append_unique_str(self.evidence_ids, *evidence_ids))

    def add_assumptions(self, *assumptions: Assumption) -> "Interpretation":
        return replace(self, assumptions=_append_assumptions(self.assumptions, *assumptions))

    def add_uncertainties(self, *uncertainties: Uncertainty) -> "Interpretation":
        return replace(self, uncertainties=_append_uncertainty(self.uncertainties, *uncertainties))

    def with_meta(self, **meta: Any) -> "Interpretation":
        return replace(self, meta=_merge_meta(self.meta, meta))

    # -----------------------
    # Convenience
    # -----------------------

    def has_provenance(self) -> bool:
        return bool(self.observation_ids) or bool(self.evidence_ids)

    def max_uncertainty_level(self) -> Optional[float]:
        if not self.uncertainties:
            return None
        return max(u.level for u in self.uncertainties)
