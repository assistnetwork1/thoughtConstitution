# constitution_providers/runner.py
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable, Tuple

from constitution_engine.models.recommendation import Recommendation, RankedOption
from constitution_engine.models.types import new_id
from constitution_engine.runtime.materialize import _normalize_violations
from constitution_engine.runtime.store import ArtifactStore

from ..context import EpisodeContext
from ..protocol import ProposalProvider
from ..protocol.proposals import ProposalSet


@dataclass(frozen=True)
class ProviderRunResult:
    """
    Output of the constitutional provider handshake.

    Notes:
    - Providers never touch the store directly.
    - Providers may not emit kernel Recommendation objects.
      They may emit Options + ranking inputs only.
    - The runner persists proposals and (optionally) attempts to materialize + validate.
    - The runner is intentionally resilient to evolving kernel entrypoints by probing
      common import paths and call signatures.
    """
    store: ArtifactStore
    proposals: ProposalSet

    episode_id: str | None = None
    orientation_id: str | None = None

    # Kernel-owned output
    canonical_recommendation_id: str | None = None

    # Provider input visibility (ranking only)
    provider_ranking_count: int = 0
    provider_rationale: str | None = None

    violations: Tuple[Any, ...] = tuple()


# -------------------------
# Store persistence helpers
# -------------------------

def _persist_one(store: ArtifactStore, obj: Any) -> None:
    for method_name in ("add", "put", "upsert", "store", "insert"):
        m = getattr(store, method_name, None)
        if callable(m):
            m(obj)
            return
    raise AttributeError(
        "ArtifactStore has no supported persistence method. "
        "Expected one of: add/put/upsert/store/insert."
    )


def _persist_many(store: ArtifactStore, items: Iterable[Any]) -> None:
    for it in items:
        _persist_one(store, it)


# -------------------------
# Canonical Recommendation build (kernel-owned)
# -------------------------

def _build_canonical_recommendation(*, ctx: EpisodeContext, proposals: ProposalSet) -> Recommendation:
    """
    Build a canonical Recommendation that is owned by the kernel boundary.

    Constitutional rule:
    - Providers do NOT emit Recommendation objects.
    - Providers emit ranking inputs only:
        proposals.proposed_ranked_options (+ optional rationale)
    - Orientation binding is canonical (ctx.orientation.orientation_id).
    - Provenance sets must only reference existing IDs (and today, providers may
      supply none; that's fine).

    Behavior:
    - Keep only RankedOptions that reference proposed options.
    - If none supplied (or all invalid), deterministic fallback ordering by option_id.
    - Re-rank to 1..N deterministically.
    """
    orientation_id = ctx.orientation.orientation_id

    # Collect proposed option IDs (kernel Option objects today)
    option_ids = tuple(opt.option_id for opt in proposals.options)
    option_id_set = set(option_ids)

    ranked: list[RankedOption] = []

    # Ranking input only
    for ro in proposals.proposed_ranked_options:
        if ro.option_id in option_id_set:
            ranked.append(ro)

    # Deterministic fallback if provider supplies no usable ranking
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

    # Normalize ranks to 1..N
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

    # NOTE: evidence/observation/interpretation IDs are empty in stub today.
    # If you later support providers emitting those artifacts as kernel objects,
    # you can safely derive IDs here.
    return Recommendation(
        recommendation_id=new_id("rc"),
        orientation_id=orientation_id,
        ranked_options=normalized_ranked,
        evidence_ids=tuple(getattr(e, "evidence_id") for e in proposals.evidence if hasattr(e, "evidence_id")),
        observation_ids=tuple(getattr(o, "observation_id") for o in proposals.observations if hasattr(o, "observation_id")),
        interpretation_ids=tuple(getattr(i, "interpretation_id") for i in proposals.interpretations if hasattr(i, "interpretation_id")),
        model_state_ids=tuple(),
        override_used=False,
        override_scope_used=tuple(),
    )


# -------------------------
# Optional kernel hooks
# -------------------------

def _try_materialize(
    store: ArtifactStore,
    ctx: EpisodeContext,
    recommendation: Recommendation,
) -> tuple[str | None, str | None]:
    """
    Attempt to materialize an episode via whatever kernel entrypoint exists.

    Returns:
      (episode_id, orientation_id)
    """
    candidates: list[tuple[str, str]] = [
        ("constitution_engine.runtime.materialize", "materialize_episode_from_recommendation"),
        ("constitution_engine.runtime.materialize", "materialize_episode"),
        ("constitution_engine.runtime.materialize", "materialize"),
        ("constitution_engine.materialize", "materialize_episode_from_recommendation"),
        ("constitution_engine.materialize", "materialize_episode"),
        ("constitution_engine.materialize", "materialize"),
    ]

    for mod_name, fn_name in candidates:
        try:
            mod = import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue

            try:
                ep = fn(
                    store=store,
                    orientation=ctx.orientation,
                    recommendation=recommendation,
                    raw_inputs=ctx.raw_inputs,
                )
            except TypeError:
                try:
                    ep = fn(store=store, orientation=ctx.orientation, recommendation=recommendation)  # type: ignore[misc]
                except TypeError:
                    ep = fn(store=store, recommendation=recommendation)  # type: ignore[misc]

            ep_id = getattr(ep, "episode_id", None) or getattr(ep, "decision_episode_id", None)
            or_id = getattr(ctx.orientation, "orientation_id", None)
            return (ep_id, or_id)
        except Exception:
            continue

    return (None, None)


def _try_validate(store: ArtifactStore, episode_id: str | None) -> Tuple[Any, ...]:
    """
    Attempt to validate the episode if validate_episode exists.
    Returns a normalized tuple of violations.
    """
    if not episode_id:
        return tuple()

    candidates: list[tuple[str, str]] = [
        ("constitution_engine.invariants.validate", "validate_episode"),
        ("constitution_engine.invariants.validate", "validate"),
        ("constitution_engine.invariants", "validate_episode"),
    ]

    for mod_name, fn_name in candidates:
        try:
            mod = import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue

            try:
                report = fn(store=store, episode_id=episode_id)  # type: ignore[misc]
            except TypeError:
                report = fn(episode_id=episode_id, store=store)  # type: ignore[misc]

            return _normalize_violations(report)
        except Exception:
            continue

    return tuple()


# -------------------------
# Public runner
# -------------------------

def run_provider(
    *,
    provider: ProposalProvider,
    ctx: EpisodeContext,
    store: ArtifactStore | None = None,
    materialize: bool = True,
    validate: bool = True,
) -> ProviderRunResult:
    """
    Execute the constitutional handshake:

      provider.propose(ctx)
        -> persist proposals into ArtifactStore
        -> build canonical Recommendation (kernel-owned)
        -> persist canonical Recommendation
        -> (optional) materialize episode
        -> (optional) validate episode

    Providers remain strictly upstream of action and cannot mutate kernel state directly.
    """
    st = store or ArtifactStore()
    proposals = provider.propose(ctx)

    # Persist proposed artifacts
    _persist_many(st, proposals.evidence)
    _persist_many(st, proposals.observations)
    _persist_many(st, proposals.interpretations)
    _persist_many(st, proposals.options)

    # Build + persist canonical recommendation (kernel-owned)
    canonical_rec = _build_canonical_recommendation(ctx=ctx, proposals=proposals)
    _persist_one(st, canonical_rec)

    ep_id: str | None = None
    or_id: str | None = None
    if materialize:
        ep_id, or_id = _try_materialize(st, ctx, canonical_rec)

    violations: Tuple[Any, ...] = tuple()
    if validate and ep_id is not None:
        violations = _try_validate(st, ep_id)

    return ProviderRunResult(
        store=st,
        proposals=proposals,
        episode_id=ep_id,
        orientation_id=or_id,
        canonical_recommendation_id=canonical_rec.recommendation_id,
        provider_ranking_count=len(proposals.proposed_ranked_options),
        provider_rationale=proposals.proposed_rationale,
        violations=violations,
    )
