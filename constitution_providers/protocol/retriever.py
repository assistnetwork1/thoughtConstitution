from __future__ import annotations

from typing import Protocol, runtime_checkable

from constitution_providers.context import EpisodeContext
from constitution_providers.protocol.proposals import DraftEnvelope


@runtime_checkable
class RetrieverProvider(Protocol):
    """
    RetrieverProviders may ONLY produce Evidence (as draft evidence).
    They may not recommend actions or assert conclusions.
    """

    def run(self, ctx: EpisodeContext) -> DraftEnvelope:
        ...
