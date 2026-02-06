from __future__ import annotations

from typing import Protocol, runtime_checkable

from constitution_providers.context import EpisodeContext
from constitution_providers.proposals import DraftEnvelope


@runtime_checkable
class ReasoningProvider(Protocol):
    """
    ReasoningProviders may propose interpretations/options/rankings as drafts.
    They may NOT choose, act, override invariants, or learn.
    """

    def run(self, ctx: EpisodeContext) -> DraftEnvelope:
        ...
