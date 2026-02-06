# constitution_providers/runner_multi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from constitution_engine.models.evidence import Evidence
from constitution_engine.models.recommendation import Recommendation, RankedOption
from constitution_engine.models.types import new_id
from constitution_engine.runtime.store import ArtifactStore

from .context import EpisodeContext
from .protocol import ProposalProvider
from .proposals import ProposalSet
from .runner import _persist_many, _persist_one, _try_materialize, _try_validate


@dataclass(frozen=True)
class ProvidersRunResult:
    """
    Multi-provider run output.

    Constitutional rule:
    - Providers emit proposals only (options + ranking inputs).
    - Kernel boundary constructs a single canonical Recommendation.
    """
    store: ArtifactStore
    proposals: Tuple[ProposalSet, ...]

    episode_id: str | None = None
    orientation_id: str | None = None
    canonical_recommendation_id: str | None = None

    violations: Tuple[Any, ...] = tuple()


def _merge_ranked_options(*, proposal_sets: Iterable[ProposalSet]) -> Tuple[RankedOption, ...]:
    """
    Deterministic merge:
    - Keep all provider RankedOptions, but dedupe by option_id (first wins).
    - Sort by (rank, option_id) for determinism.
    """
    seen: set[str] = set()
    merged: List[RankedOption] = []
    for ps in proposal_sets:
        for ro in getattr(ps, "proposed_ranked_options", ()) or ():
            if ro.option_id not in seen:
                merged.append(ro)
                seen.add(ro.option_id)
    merged_sorted = sorted(merged, key=lambda r: (r.rank, r.option_id))
    return tuple(merged_sorted)


def _build_canonical_recommendation_from_many(*, ctx: EpisodeContext, merged: ProposalSet) -> Recommendation:
    """
    Build canonical recommendation from the merged proposal set.
    (Same logic as runner._build_canonical_recommendation, but takes a merged set.)

    Constitutional rule:
    - Provider ranking is input only.
    - Canonical orientation binding comes from ctx.orientation.orientation_id.
    """
    orientation_id = ctx.orientation.orientation_id

    # Proposed option IDs (kernel model Option objects today)
    option_ids = tuple(opt.option_id for opt in merged.options)
    option_id_set = set(option_ids)

    ranked: List[RankedOption] = []
    for ro in getattr(merged, "proposed_ranked_options", ()) or ():
        if ro.option_id in option_id_set:
            ranked.append(ro)

    # Deterministic fallback if no usable ranking
    if not ranked:
        for i, opt_id in enumerate(sorted(option_ids), start=1):
            ranked.append(
                RankedOption(
                    option_id=opt_id,
                    rank=i,
                    score=0.0,
                    rationale="Default ordering (no provider ranking).",
                    confidence=None,
                    uncertainties=tuple(),
                    tradeoffs=tuple(),
                    constraint_checks=tuple(),
                )
            )

    # Normalize ranks to 1..N deterministically
    ranked_sorted = sorted(ranked, key=lambda r: (r.rank, r.option_id))
    normalized_ranked = tuple(
        RankedOption(
            option_id=ro.option_id,
            rank=i,
            score=ro.score,
            rationale=ro.rationale,
            confidence=ro.confidence,
            uncertainties=ro.uncertainties,
            tradeoffs=ro.tradeoffs,
            constraint_checks=ro.constraint_checks,
        )
        for i, ro in enumerate(ranked_sorted, start=1)
    )

    # Derive provenance ID sets only if the objects actually have those attributes.
    evidence_ids = tuple(getattr(e, "evidence_id") for e in merged.evidence if hasattr(e, "evidence_id"))
    observation_ids = tuple(getattr(o, "observation_id") for o in merged.observations if hasattr(o, "observation_id"))
    interpretation_ids = tuple(
        getattr(i, "interpretation_id") for i in merged.interpretations if hasattr(i, "interpretation_id")
    )

    return Recommendation(
        recommendation_id=new_id("rc"),
        orientation_id=orientation_id,
        ranked_options=normalized_ranked,
        evidence_ids=evidence_ids,
        observation_ids=observation_ids,
        interpretation_ids=interpretation_ids,
        model_state_ids=tuple(),
        override_used=False,
        override_scope_used=tuple(),
    )


def run_providers(
    *,
    providers: Iterable[ProposalProvider],
    ctx: EpisodeContext,
    store: ArtifactStore | None = None,
    materialize: bool = True,
    validate: bool = True,
    thread_evidence: bool = True,
) -> ProvidersRunResult:
    """
    Run multiple providers, persist their proposed artifacts, then construct ONE canonical recommendation.

    Merge policy (minimal):
    - Evidence/observations/interpretations/options are concatenated in provider order.
    - proposed_ranked_options are merged deterministically (dedupe by option_id, then sort).
    - canonical recommendation is constructed once from the merged state.

    Evidence threading (recommended):
    - If thread_evidence=True, Evidence emitted by earlier providers is passed to later
      providers via a fresh EpisodeContext snapshot. Providers still never see the store.
    """
    st = store or ArtifactStore()

    proposal_sets: List[ProposalSet] = []
    merged_evidence: List[object] = []
    merged_observations: List[object] = []
    merged_interpretations: List[object] = []
    merged_options: List[object] = []

    # Evidence "bus" passed forward constitutionally (read-only).
    evidence_bus: List[Evidence] = list(getattr(ctx, "evidence", ()) or ())

    for p in providers:
        # Provide a per-provider snapshot with accumulated evidence (if enabled)
        provider_ctx = ctx
        if thread_evidence:
            provider_ctx = EpisodeContext(
                orientation=ctx.orientation,
                raw_inputs=ctx.raw_inputs,
                evidence=tuple(evidence_bus),
                meta=ctx.meta,
            )

        ps = p.propose(provider_ctx)
        proposal_sets.append(ps)

        merged_evidence.extend(ps.evidence)
        merged_observations.extend(ps.observations)
        merged_interpretations.extend(ps.interpretations)
        merged_options.extend(ps.options)

        # Accumulate newly emitted Evidence for downstream providers
        if thread_evidence:
            for ev in ps.evidence:
                if isinstance(ev, Evidence):
                    evidence_bus.append(ev)

    merged_ranked = _merge_ranked_options(proposal_sets=proposal_sets)

    merged_rationale = " | ".join(
        [ps.proposed_rationale for ps in proposal_sets if getattr(ps, "proposed_rationale", None)]
    ) or None

    merged = ProposalSet(
        provider_id="merged",
        notes="Merged proposal sets (non-authoritative).",
        evidence=tuple(merged_evidence),
        observations=tuple(merged_observations),
        interpretations=tuple(merged_interpretations),
        options=tuple(merged_options),
        proposed_ranked_options=merged_ranked,
        proposed_rationale=merged_rationale,
    )

    # Persist all proposed artifacts
    _persist_many(st, merged.evidence)
    _persist_many(st, merged.observations)
    _persist_many(st, merged.interpretations)
    _persist_many(st, merged.options)

    # Build + persist canonical recommendation (kernel-owned)
    canonical_rec = _build_canonical_recommendation_from_many(ctx=ctx, merged=merged)
    _persist_one(st, canonical_rec)

    ep_id: str | None = None
    or_id: str | None = None
    if materialize:
        ep_id, or_id = _try_materialize(st, ctx, canonical_rec)

    violations: Tuple[Any, ...] = tuple()
    if validate and ep_id is not None:
        violations = _try_validate(st, ep_id)

    return ProvidersRunResult(
        store=st,
        proposals=tuple(proposal_sets),
        episode_id=ep_id,
        orientation_id=or_id,
        canonical_recommendation_id=canonical_rec.recommendation_id,
        violations=violations,
    )
