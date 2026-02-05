from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from .types import Reversibility, Uncertainty, Impact, new_id, now_utc


class OptionKind(str, Enum):
    EXECUTE = "execute"
    HEDGE = "hedge"
    INFO_GATHERING = "info_gathering"


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
class Option:
    """
    A possible action before ranking.

    Options must remain auditable: reference upstream artifacts.
    """
    option_id: str = field(default_factory=lambda: new_id("opt"))
    created_at: datetime = field(default_factory=now_utc)

    kind: OptionKind = OptionKind.EXECUTE
    title: str = ""
    description: str = ""

    # v0.5.1 bridge: constitutional action class gate (probe/limited/commit)
    # - None => not declared (should trigger INV-ACT-001 for EXECUTE options)
    # - expected values: "probe" | "limited" | "commit" (case-insensitive handled by rules gate)
    action_class: Optional[str] = None

    action_payload: Any = None

    orientation_id: Optional[str] = None
    observation_ids: Sequence[str] = field(default_factory=tuple)
    interpretation_ids: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)

    reversibility: Reversibility = field(default_factory=lambda: Reversibility(0.5))
    impact: Impact = field(default_factory=lambda: Impact(0.5))
    uncertainties: Sequence[Uncertainty] = field(default_factory=tuple)

    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep this permissive enough for early drafts, but never allow nonsense scalars
        # (scalar bounds are enforced in their own dataclasses).
        if self.title == "" and self.description == "":
            # Options can be created as stubs, but it’s almost always a mistake.
            # We don't raise to preserve kernel neutrality; invariants can enforce later.
            pass

    # -----------------------
    # Immutability helpers
    # -----------------------

    def with_title(self, title: str) -> "Option":
        return replace(self, title=title)

    def with_description(self, description: str) -> "Option":
        return replace(self, description=description)

    def with_kind(self, kind: OptionKind) -> "Option":
        return replace(self, kind=kind)

    def with_action_class(self, action_class: Optional[str]) -> "Option":
        """
        v0.5.1 bridge field setter.
        - None => not declared (should trigger INV-ACT-001 for EXECUTE options)
        - expected string values: "probe" | "limited" | "commit"
        """
        return replace(self, action_class=action_class)

    def with_action_payload(self, payload: Any) -> "Option":
        return replace(self, action_payload=payload)

    def with_orientation(self, orientation_id: Optional[str]) -> "Option":
        return replace(self, orientation_id=orientation_id)

    def with_reversibility(self, reversibility: Reversibility) -> "Option":
        return replace(self, reversibility=reversibility)

    def with_impact(self, impact: Impact) -> "Option":
        return replace(self, impact=impact)

    def add_observations(self, *observation_ids: str) -> "Option":
        return replace(self, observation_ids=_append_unique_str(self.observation_ids, *observation_ids))

    def add_interpretations(self, *interpretation_ids: str) -> "Option":
        return replace(self, interpretation_ids=_append_unique_str(self.interpretation_ids, *interpretation_ids))

    def add_evidence(self, *evidence_ids: str) -> "Option":
        return replace(self, evidence_ids=_append_unique_str(self.evidence_ids, *evidence_ids))

    def add_uncertainties(self, *uncertainties: Uncertainty) -> "Option":
        return replace(self, uncertainties=_append_uncertainty(self.uncertainties, *uncertainties))

    def with_meta(self, **meta: Any) -> "Option":
        return replace(self, meta=_merge_meta(self.meta, meta))

    # -----------------------
    # Convenience
    # -----------------------

    def has_upstream_references(self) -> bool:
        """
        Kernel-friendly check for auditability.
        Not enforced here (apps may draft options early), but useful for invariants.
        """
        return bool(self.observation_ids) or bool(self.interpretation_ids) or bool(self.evidence_ids)

    def max_uncertainty_level(self) -> Optional[float]:
        if not self.uncertainties:
            return None
        return max(u.level for u in self.uncertainties)
