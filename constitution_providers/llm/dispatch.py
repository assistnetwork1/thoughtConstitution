# constitution_providers/llm/dispatch.py
from __future__ import annotations

from typing import Mapping

from constitution_engine.invariants.provider_rules import validate_proposalset

from constitution_providers.context import EpisodeContext
from constitution_providers.protocol.proposals import ProposalSet

from constitution_providers.llm.packing import RenderedPrompt, render_prompt
from constitution_providers.llm.registry import (
    DEFAULT_ADAPTERS,
    DEFAULT_PACKS,
    AdapterRegistry,
    ModelRouteSpec,
    PackRegistry,
)


def _build_evidence_by_id(ctx: EpisodeContext) -> dict[str, object]:
    """
    Build evidence_id -> Evidence object map from the EpisodeContext.

    This is the authoritative resolution set for INV-PS-002
    (provider evidence_refs must resolve).
    """
    out: dict[str, object] = {}
    for ev in getattr(ctx, "evidence", ()) or ():
        ev_id = getattr(ev, "evidence_id", None)
        if isinstance(ev_id, str) and ev_id.strip():
            out[ev_id] = ev
    return out


def dispatch(
    *,
    ctx: EpisodeContext,
    route: ModelRouteSpec,
    packs: PackRegistry = DEFAULT_PACKS,
    adapters: AdapterRegistry = DEFAULT_ADAPTERS,
) -> ProposalSet:
    """
    Stable LLM spine:

      pack -> render_prompt -> adapter.invoke -> adapter.parse_to_proposalset -> provider_rules

    Notes:
    - No policy here.
    - No heuristics here.
    - Deterministic wiring + hard constitutional boundary validation.
    """
    pack = packs.get(route.pack_id)
    adapter = adapters.get(route.adapter_key)

    extra: dict[str, object] = {
        "provider_id": route.provider_id,
        "model_id": route.model_id,
        "adapter_key": route.adapter_key,
        "pack_id": route.pack_id,
        "temperature": route.temperature,
        "limits": route.limits,
        "provider_version": route.provider_version,
    }

    # route.extra_meta should be Mapping[str, object] by type, but normalize defensively
    if isinstance(route.extra_meta, Mapping):
        extra.update(dict(route.extra_meta))

    rendered: RenderedPrompt = render_prompt(pack=pack, ctx=ctx, extra=extra)

    payload = adapter.invoke(
        rendered_prompt=rendered,
        model_id=route.model_id,
        temperature=route.temperature,
    )

    ps = adapter.parse_to_proposalset(
        payload=payload,
        ctx=ctx,
        provider_id=route.provider_id,
        model_id=route.model_id,
        limits=route.limits,
        temperature=route.temperature,
        provider_version=route.provider_version,
        prompt_meta=rendered.meta,
    )

    # -------------------------
    # Hard boundary enforcement
    # -------------------------
    evidence_by_id = _build_evidence_by_id(ctx)
    violations = validate_proposalset(ps, evidence_by_id=evidence_by_id)
    if violations:
        msg = "; ".join(f"{v.rule}: {v.message}" for v in violations)
        raise ValueError(f"ProposalSet failed provider_rules: {msg}")

    return ps
