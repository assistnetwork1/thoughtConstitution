# constitution_providers/llm/llm_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from constitution_providers.context import EpisodeContext
from constitution_providers.protocol import ProposalProvider
from constitution_providers.protocol.proposals import ProposalSet

from constitution_providers.llm.openai.packing import (
    PromptPack,
    RenderedPrompt,
    default_reasoner_pack_v1,
    render_prompt,
)


@dataclass(frozen=True)
class LLMRouteSpec:
    """
    Declarative routing spec:
      - provider_id: stable provider identity for audit trails
      - model_id: model name/alias (e.g. "gpt-4.1-mini", "o4-mini", etc.)
      - pack: PromptPack that defines how we render context + instructions
      - temperature: sampling temperature (threaded into ProposalSet.sampling)
    """
    provider_id: str
    model_id: str
    pack: PromptPack
    temperature: float = 0.0


class LLMProvider(ProposalProvider):
    """
    Model-agnostic provider:

        EpisodeContext
          -> render_prompt(pack, ctx)
          -> adapter.invoke(rendered_prompt, model_id, temperature)
          -> adapter.parse_to_proposalset(...)
          -> ProposalSet

    Constitutional boundary:
    - This provider never constructs kernel-owned artifacts.
    - It delegates schema enforcement of ProposalSet content to adapter.parse_to_proposalset.
    - It does not score, rank, or "think" here; it's a transport + packing wrapper.

    Adapter contract (required methods):
      - invoke(rendered_prompt: RenderedPrompt, model_id: str, temperature: float) -> Any
      - parse_to_proposalset(
            payload: Any,
            ctx: EpisodeContext,
            provider_id: str,
            model_id: str,
            limits: str,
            temperature: float,
            provider_version: str,
            prompt_meta: Mapping[str, object],
        ) -> ProposalSet
    """

    # Protocol attribute (stable identifier)
    provider_id: str

    def __init__(
        self,
        *,
        route: LLMRouteSpec,
        adapter: Any,
        provider_version: str = "v1",
        limits: str = "stub-limits",
        extra_meta: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.provider_id = route.provider_id
        self._route = route
        self._adapter = adapter
        self._provider_version = provider_version
        self._limits = limits
        self._extra_meta = dict(extra_meta or {})

    def propose(self, ctx: EpisodeContext) -> ProposalSet:
        # 1) Pack/Render prompt deterministically
        rendered: RenderedPrompt = render_prompt(
            pack=self._route.pack,
            ctx=ctx,
            extra={
                "provider_id": self.provider_id,
                "model_id": self._route.model_id,
                "temperature": self._route.temperature,
                **self._extra_meta,
            },
        )

        # 2) Invoke model via adapter (adapter owns SDK / HTTP specifics)
        payload = self._adapter.invoke(
            rendered_prompt=rendered,
            model_id=self._route.model_id,
            temperature=self._route.temperature,
        )

        # 3) Adapter must return a provider_rules-compatible ProposalSet
        return self._adapter.parse_to_proposalset(
            payload=payload,
            ctx=ctx,
            provider_id=self.provider_id,
            model_id=self._route.model_id,
            limits=self._limits,
            temperature=self._route.temperature,
            provider_version=self._provider_version,
            prompt_meta=rendered.meta,
        )


def default_openai_reasoner_provider(
    *,
    model_id: str,
    adapter: Any,
    temperature: float = 0.0,
    provider_id: str = "openai_reasoner",
) -> LLMProvider:
    """
    Convenience factory:
    - Uses the default_reasoner_pack_v1() pack
    - Leaves adapter selection to caller (OpenAIAdapter, etc.)
    """
    route = LLMRouteSpec(
        provider_id=provider_id,
        model_id=model_id,
        pack=default_reasoner_pack_v1(),
        temperature=temperature,
    )
    return LLMProvider(route=route, adapter=adapter)
