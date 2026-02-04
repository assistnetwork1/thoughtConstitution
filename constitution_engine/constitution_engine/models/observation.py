from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Sequence, Tuple, Optional

from .types import InfoType, Confidence, Uncertainty, new_id, now_utc


_OBSERVATIONAL_TYPES = {
    InfoType.FACT,
    InfoType.MEASUREMENT,
    InfoType.EVENT,
    InfoType.TESTIMONY,
}


def _as_tuple_str(seq: Sequence[str]) -> Tuple[str, ...]:
    return tuple(seq)


def _as_tuple_unc(seq: Sequence[Uncertainty]) -> Tuple[Uncertainty, ...]:
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


def _merge_meta(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Mapping[str, Any]:
    if not patch:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


@dataclass(frozen=True)
class Observation:
    """
    Reality-anchored statement(s) derived from evidence, explicitly typed.
    Observations must be observational InfoTypes only.
    """
    observation_id: str = field(default_factory=lambda: new_id("obs"))
    created_at: datetime = field(default_factory=now_utc)

    info_type: InfoType = InfoType.FACT
    statement: str = ""
    data: Any = None

    raw_input_ids: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)

    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    tags: Sequence[str] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.info_type not in _OBSERVATIONAL_TYPES:
            raise ValueError(
                f"Observation.info_type must be observational; got {self.info_type}"
            )

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_statement(self, statement: str, *, data: Any = None) -> "Observation":
        return replace(self, statement=statement, data=data if data is not None else self.data)

    def with_confidence(self, confidence: Confidence) -> "Observation":
        return replace(self, confidence=confidence)

    def add_raw_inputs(self, *raw_input_ids: str) -> "Observation":
        return replace(self, raw_input_ids=_append_unique_str(self.raw_input_ids, *raw_input_ids))

    def add_evidence(self, *evidence_ids: str) -> "Observation":
        return replace(self, evidence_ids=_append_unique_str(self.evidence_ids, *evidence_ids))

    def add_uncertainties(self, *uncertainties: Uncertainty) -> "Observation":
        return replace(self, uncertainties=_append_uncertainty(self.uncertainties, *uncertainties))

    def add_tags(self, *tags: str) -> "Observation":
        return replace(self, tags=_append_unique_str(self.tags, *tags))

    def with_meta(self, **meta: Any) -> "Observation":
        return replace(self, meta=_merge_meta(self.meta, meta))

    # -----------------------
    # Convenience checks
    # -----------------------

    def has_provenance(self) -> bool:
        """
        Kernel-friendly check: observational claims should usually cite raw inputs and/or evidence.
        Not enforced here (apps may create placeholder observations), but useful for invariants.
        """
        return bool(self.raw_input_ids) or bool(self.evidence_ids)

    def max_uncertainty_level(self) -> Optional[float]:
        if not self.uncertainties:
            return None
        return max(u.level for u in self.uncertainties)
