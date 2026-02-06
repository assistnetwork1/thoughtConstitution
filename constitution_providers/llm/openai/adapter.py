# constitution_providers/llm/openai/adapter.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Mapping

from constitution_providers.context import EpisodeContext
from constitution_providers.protocol.proposals import (
    ProposalSampling,
    ProposalSet,
    ProposalUncertainty,
    ProposedInterpretation,
    ProposedOption,
    ProposedRankedOption,
    OverrideSuggestion,
)

from .packing import RenderedPrompt
from .client import OpenAIClient, OpenAIRequest


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _as_tuple_str(x: Any) -> tuple[str, ...]:
    if x is None:
        return ()
    if isinstance(x, (list, tuple)):
        return tuple(str(i) for i in x if str(i).strip())
    return (str(x),)


def _get(d: Mapping[str, Any], k: str, default: Any = None) -> Any:
    return d.get(k, default)


@dataclass(frozen=True)
class OpenAIAdapter:
    """
    JSON-only adapter:
      RenderedPrompt -> OpenAIClient -> JSON -> ProposalSet

    This adapter:
    - does NOT do scoring/heuristics
    - only parses + canonicalizes to provider_rules-compatible ProposalSet
    """

    client: OpenAIClient

    def invoke(self, *, rendered_prompt: RenderedPrompt, model_id: str, temperature: float) -> Any:
        """
        Transport call only. Returns raw payload (str/dict/etc).
        Parsing happens in parse_to_proposalset().
        """
        req = OpenAIRequest(
            rendered_prompt=rendered_prompt,
            model_id=model_id,
            temperature=float(temperature),
        )
        return self.client.invoke(req)

    def parse_to_proposalset(
        self,
        *,
        payload: Any,
        ctx: EpisodeContext,
        provider_id: str,
        model_id: str,
        limits: str,
        temperature: float,
        provider_version: str,
        prompt_meta: Mapping[str, object],
    ) -> ProposalSet:
        # ---- Payload normalization (strict but robust) ----
        if isinstance(payload, str):
            data = json.loads(payload)
        elif isinstance(payload, Mapping):
            data = dict(payload)
        elif is_dataclass(payload):
            data = asdict(payload)
        else:
            raise TypeError("OpenAIAdapter payload must be JSON string or mapping.")

        # ---- Interpretations ----
        interpretations: list[ProposedInterpretation] = []
        for it in data.get("interpretations", []) or []:
            if not isinstance(it, Mapping):
                continue
            unc = _get(it, "uncertainty", {}) or {}
            interpretations.append(
                ProposedInterpretation(
                    interpretation_id=str(_get(it, "interpretation_id", "")),
                    info_type=str(_get(it, "info_type", "hypothesis")),
                    text=str(_get(it, "text", "")),
                    confidence=_clamp01(float(_get(it, "confidence", 0.5))),
                    uncertainty=ProposalUncertainty(
                        level=_clamp01(float(_get(unc, "level", 0.5)))
                    ),
                    evidence_refs=_as_tuple_str(_get(it, "evidence_refs", ())),
                    limits=str(_get(it, "limits", limits)),
                    meta=dict(_get(it, "meta", {}) or {}),
                )
            )

        # ---- Options ----
        options: list[ProposedOption] = []
        for op in data.get("options", []) or []:
            if not isinstance(op, Mapping):
                continue
            unc = _get(op, "uncertainty", {}) or {}
            options.append(
                ProposedOption(
                    option_id=str(_get(op, "option_id", "")),
                    kind=str(_get(op, "kind", "execute")),
                    title=str(_get(op, "title", "")),
                    description=str(_get(op, "description", "")),
                    action_class=str(_get(op, "action_class", "probe")),
                    impact=_clamp01(float(_get(op, "impact", 0.3))),
                    reversibility=_clamp01(float(_get(op, "reversibility", 0.8))),
                    confidence=_clamp01(float(_get(op, "confidence", 0.5))),
                    uncertainty=ProposalUncertainty(
                        level=_clamp01(float(_get(unc, "level", 0.5)))
                    ),
                    evidence_refs=_as_tuple_str(_get(op, "evidence_refs", ())),
                    limits=str(_get(op, "limits", limits)),
                    meta=dict(_get(op, "meta", {}) or {}),
                )
            )

        # ---- Ranked Options ----
        ranked: list[ProposedRankedOption] = []
        for ro in data.get("ranked_options", []) or []:
            if not isinstance(ro, Mapping):
                continue
            unc = _get(ro, "uncertainty", {}) or {}
            title_val = _get(ro, "title", None)
            ranked.append(
                ProposedRankedOption(
                    rank=int(_get(ro, "rank", 1)),
                    option_ref=str(_get(ro, "option_ref", "")),
                    rationale=str(_get(ro, "rationale", "")),
                    title=None if title_val is None else str(title_val),
                    confidence=_clamp01(float(_get(ro, "confidence", 0.5))),
                    uncertainty=ProposalUncertainty(
                        level=_clamp01(float(_get(unc, "level", 0.5)))
                    ),
                    evidence_refs=_as_tuple_str(_get(ro, "evidence_refs", ())),
                    limits=str(_get(ro, "limits", limits)),
                    meta=dict(_get(ro, "meta", {}) or {}),
                )
            )

        # ---- Override Suggestions (non-executable only) ----
        overrides: list[OverrideSuggestion] = []
        for ov in data.get("override_suggestions", []) or []:
            if not isinstance(ov, Mapping):
                continue
            unc = _get(ov, "uncertainty", {}) or {}
            overrides.append(
                OverrideSuggestion(
                    invariant_id=str(_get(ov, "invariant_id", "")),
                    reason=str(_get(ov, "reason", "")),
                    scope=str(_get(ov, "scope", "")),
                    confidence=_clamp01(float(_get(ov, "confidence", 0.5))),
                    uncertainty=ProposalUncertainty(
                        level=_clamp01(float(_get(unc, "level", 0.5)))
                    ),
                    evidence_refs=_as_tuple_str(_get(ov, "evidence_refs", ())),
                    limits=str(_get(ov, "limits", limits)),
                    meta=dict(_get(ov, "meta", {}) or {}),
                )
            )

        # ---- Run ID ----
        ori_id = getattr(getattr(ctx, "orientation", None), "orientation_id", "ori_unknown")
        run_id = (
            f"run:{provider_id}:{ori_id}:"
            f"{prompt_meta.get('pack_id','pack')}:{prompt_meta.get('pack_version','v1')}"
        )

        return ProposalSet(
            provider_id=provider_id,
            model_id=model_id,
            run_id=run_id,
            sampling=ProposalSampling(temperature=float(temperature)),
            limits=limits,
            meta={
                "provider_version": provider_version,
                "prompt_meta": dict(prompt_meta),
            },
            interpretations=tuple(interpretations),
            options=tuple(options),
            ranked_options=tuple(ranked),
            override_suggestions=tuple(overrides),
        )
