# constitution_providers/llm/openai/client_sdk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .client import OpenAIClient, OpenAIRequest


@dataclass(frozen=True)
class OpenAISDKConfig:
    """
    Minimal config for the OpenAI Python SDK client.
    Keep it transport-only: no schema, no parsing, no policy.
    """
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None


class OpenAISDKClient(OpenAIClient):
    """
    Real transport client using the OpenAI Python SDK (Responses API).

    Contract:
      invoke(OpenAIRequest) -> raw payload (we return text, expected to be JSON-only per prompt pack)
    """

    def __init__(self, *, config: OpenAISDKConfig | None = None) -> None:
        cfg = config or OpenAISDKConfig()

        # The OpenAI() constructor pulls from env by default; these kwargs override when provided.
        kwargs: dict[str, Any] = {}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.organization:
            kwargs["organization"] = cfg.organization
        if cfg.project:
            kwargs["project"] = cfg.project

        self._client = OpenAI(**kwargs)

    def invoke(self, req: OpenAIRequest) -> Any:
        """
        Use Responses API:
          - instructions = system message
          - input = user message
        Return aggregated output text (SDK convenience).
        """
        resp = self._client.responses.create(
            model=req.model_id,
            instructions=req.rendered_prompt.system,
            input=req.rendered_prompt.user,
            temperature=req.temperature,
        )

        # Prefer SDK convenience property when available; it aggregates output text items.
        out = getattr(resp, "output_text", None)
        if isinstance(out, str):
            return out

        # Defensive fallback: return the raw response object / dict-ish payload.
        # Adapter can decide how to handle it (but our OpenAIAdapter expects str or Mapping).
        return resp
