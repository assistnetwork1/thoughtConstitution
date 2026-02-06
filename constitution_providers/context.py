# constitution_providers/context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

from constitution_engine.models.orientation import Orientation
from constitution_engine.models.raw_input import RawInput
from constitution_engine.models.evidence import Evidence


def _as_tuple(seq: Sequence[object]) -> Tuple[object, ...]:
    return tuple(seq)


@dataclass(frozen=True)
class EpisodeContext:
    """
    Minimal, read-only provider context.

    Constitutional constraints:
    - Providers receive only what the kernel (or caller) explicitly supplies.
    - Providers do not receive write access to the ArtifactStore.
    - Immutable + deterministic (tuples) for audit/logging.

    Canonical anchors:
    - Orientation: goals / constraints / values / risk posture
    - Evidence: provenance anchors retrieved upstream
    """

    orientation: Orientation
    raw_inputs: Tuple[RawInput, ...] = ()
    evidence: Tuple[Evidence, ...] = ()

    # Optional: extra, caller-supplied metadata for providers (never authoritative)
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_inputs", _as_tuple(self.raw_inputs))   # type: ignore[arg-type]
        object.__setattr__(self, "evidence", _as_tuple(self.evidence))       # type: ignore[arg-type]
        object.__setattr__(self, "meta", dict(self.meta) if self.meta else {})
