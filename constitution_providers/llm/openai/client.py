# constitution_providers/llm/openai/client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from constitution_providers.llm.openai.packing import RenderedPrompt


@dataclass(frozen=True)
class OpenAIRequest:
    """
    Transport-layer request for an OpenAI-backed adapter.

    Note:
    - rendered_prompt contains the fully packed system/user content + metadata
    - model_id + temperature are threaded explicitly for audit + replay
    """
    rendered_prompt: RenderedPrompt
    model_id: str
    temperature: float = 0.0


@runtime_checkable
class OpenAIClient(Protocol):
    """
    Minimal client interface expected by OpenAIAdapter.

    This is intentionally tiny:
    - adapter owns schema + parsing
    - client owns transport (SDK/HTTP) and returns raw payloads
    """

    def invoke(self, req: OpenAIRequest) -> Any:
        """
        Execute a model call and return the raw payload (dict/str/etc).
        Adapter will parse this into ProposalSet.
        """
        ...


class OpenAIClientStub(OpenAIClient):
    """
    Deterministic stub for tests.

    Supports multiple constructor aliases to stay compatible with evolving tests:

      OpenAIClientStub(payload=<any>)
      OpenAIClientStub(payloads=[<any>, <any>])
      OpenAIClientStub(fixture_json=<any>)   # alias of payload
      OpenAIClientStub(fixtures_json=[...])  # alias of payloads

    Returns sequentially; if invoked more times than provided, repeats last payload.
    """

    def __init__(
        self,
        *,
        payload: Any | None = None,
        payloads: list[Any] | None = None,
        fixture_json: Any | None = None,
        fixtures_json: list[Any] | None = None,
    ) -> None:
        # Normalize aliases first
        if payloads is None and fixtures_json is not None:
            payloads = list(fixtures_json)

        if payload is None and fixture_json is not None:
            payload = fixture_json

        if payloads is not None:
            self._payloads = list(payloads)
        else:
            # single payload (can be None)
            self._payloads = [payload]

        self._i = 0

    def invoke(self, req: OpenAIRequest) -> Any:
        if not self._payloads:
            return None
        if self._i >= len(self._payloads):
            return self._payloads[-1]
        out = self._payloads[self._i]
        self._i += 1
        return out
