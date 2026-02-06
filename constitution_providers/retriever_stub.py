# constitution_providers/retriever_stub.py
from __future__ import annotations

from constitution_engine.models.evidence import Evidence, SourceRef, SpanRef
from constitution_engine.models.types import Confidence

from .context import EpisodeContext
from .protocol.proposals import ProposalSet
from .protocol import ProposalProvider


class StubRetrieverProvider(ProposalProvider):
    """
    Minimal retriever provider:
      - emits Evidence only (no assertions, no options, no ranking)

    Constitutional note:
      Evidence is a provenance anchor, NOT a truth claim.
    """

    provider_id = "stub_retriever"
    capabilities = frozenset({"evidence"})

    def propose(self, ctx: EpisodeContext) -> ProposalSet:
        # Use the first raw input as the "retrieved" content for now (pure stub).
        raw_text = ""
        if ctx.raw_inputs:
            raw_text = str(ctx.raw_inputs[0].payload)

        src = SourceRef(
            uri="stub://local/raw_input",
            title="Raw input (stub retriever)",
            author=None,
            published_at=None,
            extra={"mode": "stub_retriever"},
        )

        spans = (SpanRef(start=0, end=len(raw_text)),) if raw_text else tuple()

        ev = Evidence(
            sources=(src,),
            spans=spans,
            summary="Stub-retrieved raw input payload (provenance only).",
            notes={"raw_input_preview": raw_text[:200], "mode": "stub_retriever"},
            integrity=Confidence(1.0),
        )

        return ProposalSet(
            provider_id=self.provider_id,
            notes="StubRetriever: emits evidence only.",
            evidence=(ev,),
            observations=tuple(),
            interpretations=tuple(),
            options=tuple(),
            proposed_ranked_options=tuple(),
            proposed_rationale=None,
        )
