# constitution_providers/drafter_shim.py
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Iterable, Tuple, cast

from constitution_engine.intake.adapter import Drafter, draft_episode
from constitution_engine.models.evidence import Evidence
from constitution_engine.models.interpretation import Interpretation
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option
from constitution_engine.models.recommendation import Recommendation

from .context import EpisodeContext
from .protocol.proposals import ProposalSet
from .protocol import ProposalProvider


def _as_tuple(items: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(items) if items else tuple()


def _get(obj: Any, name: str, default: Any) -> Any:
    """
    Defensive attribute/dict access:
      - supports dataclasses/objects with attributes
      - supports dict-like payloads
    """
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


class DrafterShim(ProposalProvider):
    """
    Adapter: treat the existing intake Drafter/draft_episode as a ProposalProvider.

    Constitutional posture:
    - The drafter produces a *draft* bundle (advisory artifacts).
    - This shim simply re-exports those artifacts as a ProposalSet.
    - No store mutation. No choose/act. No validation here.

    This lets you plug the current intake pipeline into the provider ecosystem
    without rewriting the drafter logic.
    """

    provider_id = "drafter_shim"
    capabilities = frozenset({"evidence", "observation", "interpretation", "option", "recommendation"})

    def __init__(self, drafter: Drafter | None = None) -> None:
        # Accept injected drafter for testing/experiments; default to canonical Drafter.
        self._drafter = drafter or Drafter()

    def propose(self, ctx: EpisodeContext) -> ProposalSet:
        # Canonical contract (current codebase posture):
        # draft_episode(*, goal=?, raw_inputs=[...], ...) -> draft bundle
        #
        # We no longer assume a separate GoalSpec exists. Orientation is the "goal container".
        # Some adapters may accept `orientation=` directly; others may still use `goal=`.
        try:
            draft = draft_episode(orientation=ctx.orientation, raw_inputs=list(ctx.raw_inputs))  # type: ignore[call-arg]
        except TypeError:
            # Back-compat: older draft_episode signature may still use goal=
            draft = draft_episode(goal=ctx.orientation, raw_inputs=list(ctx.raw_inputs))  # type: ignore[call-arg]

        # The draft may be a dataclass, object, or dict — extract defensively.
        evidence = cast(
            Tuple[Evidence, ...],
            _as_tuple(_get(draft, "evidence_items", _get(draft, "evidence", ()))),
        )
        observations = cast(Tuple[Observation, ...], _as_tuple(_get(draft, "observations", ())))
        interpretations = cast(
            Tuple[Interpretation, ...],
            _as_tuple(_get(draft, "interpretations", ())),
        )
        options = cast(Tuple[Option, ...], _as_tuple(_get(draft, "options", ())))
        recommendation = cast(Recommendation | None, _get(draft, "recommendation", None))

        # Some drafter variants might return a wrapper with .bundle or similar.
        # If we got nothing at the top level, attempt one more unwrap.
        if not (evidence or observations or interpretations or options or recommendation):
            inner = _get(draft, "bundle", None)
            if inner is not None:
                evidence = cast(
                    Tuple[Evidence, ...],
                    _as_tuple(_get(inner, "evidence_items", _get(inner, "evidence", ()))),
                )
                observations = cast(Tuple[Observation, ...], _as_tuple(_get(inner, "observations", ())))
                interpretations = cast(
                    Tuple[Interpretation, ...],
                    _as_tuple(_get(inner, "interpretations", ())),
                )
                options = cast(Tuple[Option, ...], _as_tuple(_get(inner, "options", ())))
                recommendation = cast(Recommendation | None, _get(inner, "recommendation", None))

        return ProposalSet(
            provider_id=self.provider_id,
            notes="DrafterShim: exported draft_episode artifacts as advisory proposals.",
            meta={
                "draft_type": type(draft).__name__,
                "is_dataclass": is_dataclass(draft),
            },
            evidence=evidence,
            observations=observations,
            interpretations=interpretations,
            options=options,
            recommendation=recommendation,
        )
