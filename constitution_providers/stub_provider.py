# constitution_providers/stub_provider.py
from __future__ import annotations

from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import RankedOption
from constitution_engine.models.types import (
    Confidence,
    Impact,
    Reversibility,
    Uncertainty,
    new_id,
)

from .context import EpisodeContext
from .proposals import ProposalSet
from .protocol import ProposalProvider


class StubProvider(ProposalProvider):
    """
    Minimal provider:
      - proposes a single PROBE option (info gathering)
      - proposes it as rank=1 (ranking input only)

    Notes:
    - This is intentionally "safe default" behavior.
    - It does NOT mutate the store, choose, or act.
    - The kernel constructs the canonical Recommendation after gating/constraints.
    """

    provider_id = "stub_provider"
    capabilities = frozenset({"option", "ranking"})

    def propose(self, ctx: EpisodeContext) -> ProposalSet:
        orientation_id = ctx.orientation.orientation_id

        # Evidence-aware (constitutional):
        # Providers can READ evidence passed via ctx, but cannot touch the store.
        evidence_ids = tuple(ev.evidence_id for ev in getattr(ctx, "evidence", ()) or ())

        # Slightly adjust confidence based on whether we have any provenance anchors
        base_conf = 0.75 if evidence_ids else 0.65

        opt = Option(
            option_id=new_id("op"),
            kind=OptionKind.INFO_GATHERING,  # safest default
            title="Gather missing information",
            description="Probe: request/collect the information needed to decide safely.",
            orientation_id=orientation_id,
            impact=Impact(value=0.1),
            reversibility=Reversibility(value=0.95),
            uncertainties=(Uncertainty(level=0.2, description="Probing is low-risk."),),
            action_class="probe",
            observation_ids=tuple(),
            interpretation_ids=tuple(),
            evidence_ids=evidence_ids,  # ✅ link option to retrieved evidence
        )

        ro = RankedOption(
            option_id=opt.option_id,
            rank=1,
            score=0.9,
            rationale=(
                "Default-safe probe when information is incomplete. "
                + ("(Evidence available.)" if evidence_ids else "(No evidence provided.)")
            ),
            confidence=Confidence(value=base_conf),
            uncertainties=(Uncertainty(level=0.2, description="Probe is low-risk."),),
            tradeoffs=tuple(),
            constraint_checks=tuple(),
        )

        return ProposalSet(
            provider_id=self.provider_id,
            notes="StubProvider: safe-default PROBE ranking (evidence-aware).",
            evidence=tuple(),
            observations=tuple(),
            interpretations=tuple(),
            options=(opt,),
            proposed_ranked_options=(ro,),
            proposed_rationale=ro.rationale,
        )
