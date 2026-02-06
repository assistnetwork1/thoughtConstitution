# constitution_providers/llm/registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Protocol, runtime_checkable

from constitution_providers.llm.packing import PromptPack


# ----------------------------
# Route spec (declarative)
# ----------------------------

@dataclass(frozen=True)
class ModelRouteSpec:
    """
    Declarative routing:
    - selects adapter backend via adapter_key
    - selects prompt pack via pack_id
    - threads model_id + sampling + limits into ProposalSet (via adapter.parse_to_proposalset)
    """
    model_id: str
    provider_id: str
    adapter_key: str
    pack_id: str

    temperature: float = 0.0
    limits: str = "stub-limits"
    provider_version: str = "v1"

    # Optional meta threaded into prompt rendering + adapter meta.
    # Must be mapping-like; keep default empty dict via default_factory.
    extra_meta: Mapping[str, object] = field(default_factory=dict)


# ----------------------------
# Adapter factory registry
# ----------------------------

@runtime_checkable
class LLMAdapter(Protocol):
    """
    Adapter contract (backend-specific transport + parsing).

    Must implement:
      - invoke(rendered_prompt, model_id, temperature) -> Any payload
      - parse_to_proposalset(payload, ctx, provider_id, model_id, limits, temperature, provider_version, prompt_meta) -> ProposalSet
    """

    def invoke(self, *, rendered_prompt: Any, model_id: str, temperature: float) -> Any: ...

    def parse_to_proposalset(
        self,
        *,
        payload: Any,
        ctx: Any,
        provider_id: str,
        model_id: str,
        limits: str,
        temperature: float,
        provider_version: str,
        prompt_meta: Mapping[str, object],
    ) -> Any: ...


AdapterFactory = Callable[[], LLMAdapter]


class AdapterRegistry:
    """
    adapter_key -> adapter factory

    We store factories (not instances) so call-sites can stay deterministic and avoid
    shared mutable adapter state unless you explicitly want it.
    """

    def __init__(self) -> None:
        self._factories: MutableMapping[str, AdapterFactory] = {}

    def register(self, key: str, factory: AdapterFactory) -> None:
        k = str(key).strip()
        if not k:
            raise ValueError("adapter registry key must be non-empty")
        if not callable(factory):
            raise TypeError("adapter registry factory must be callable")
        self._factories[k] = factory

    def get(self, key: str) -> LLMAdapter:
        k = str(key).strip()
        if k not in self._factories:
            raise KeyError(f"Unknown adapter_key: {k}")
        return self._factories[k]()  # instantiate on demand

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))


# ----------------------------
# Prompt pack registry
# ----------------------------

class PackRegistry:
    """
    pack_id -> PromptPack
    """

    def __init__(self) -> None:
        self._packs: MutableMapping[str, PromptPack] = {}

    def register(self, pack: PromptPack) -> None:
        pid = str(pack.pack_id).strip()
        if not pid:
            raise ValueError("PromptPack.pack_id must be non-empty")
        self._packs[pid] = pack

    def get(self, pack_id: str) -> PromptPack:
        pid = str(pack_id).strip()
        if pid not in self._packs:
            raise KeyError(f"Unknown pack_id: {pid}")
        return self._packs[pid]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._packs.keys()))


# ----------------------------
# Optional: default singleton registries
# ----------------------------

DEFAULT_PACKS = PackRegistry()
DEFAULT_ADAPTERS = AdapterRegistry()
