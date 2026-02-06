# constitution_providers/protocol.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProposalProvider(Protocol):
    """
    A constitutional provider is anything "smart" that lives OUTSIDE the kernel:
    - LLMs / symbolic reasoners / planners
    - RAG / search / API retrievers
    - rules engines
    - simulators

    Constitutional constraints:
    - Providers can only PROPOSE artifacts (Evidence/Observation/Interpretation/Option/etc.).
    - Providers cannot choose/act.
    - Providers cannot mutate kernel state or the ArtifactStore.
    - Providers must be deterministic given the same inputs OR declare non-determinism
      via metadata inside the ProposalSet they emit (preferred: include a run_id/seed).
    """

    # A stable identifier for audit trails, attribution, and calibration.
    # (Protocol attribute: implementing classes should define it.)
    provider_id: str

    def propose(self, ctx: "EpisodeContext") -> "ProposalSet":
        """
        Generate a ProposalSet in response to an EpisodeContext.

        The returned ProposalSet is purely advisory: the kernel will normalize,
        persist, and validate proposals through its constitutional pipeline.
        """
        ...


@runtime_checkable
class RetrieverProvider(ProposalProvider, Protocol):
    """
    Retriever specialization.

    Expected output: primarily Evidence proposals (and optionally "candidate observation"
    proposals that remain reality-typed and evidence-linked).
    """

    # e.g., {"evidence"} or {"evidence", "observation_candidates"}
    capabilities: frozenset[str]

    def propose(self, ctx: "EpisodeContext") -> "ProposalSet":
        ...


@runtime_checkable
class ReasoningProvider(ProposalProvider, Protocol):
    """
    Reasoning specialization.

    Expected output: interpretations, options, and optionally ranked-option proposals.
    Never actions. Never choices.
    """

    # e.g., {"interpretation", "option", "ranking"}
    capabilities: frozenset[str]

    def propose(self, ctx: "EpisodeContext") -> "ProposalSet":
        ...


# Local imports at bottom to avoid import cycles at import time.
from .context import EpisodeContext
from .proposals import ProposalSet
