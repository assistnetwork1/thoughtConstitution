# constitution_engine/tests/test_llm_dispatch_smoke.py
from __future__ import annotations

from typing import Any, Mapping

from constitution_engine.invariants.provider_rules import validate_proposalset
from constitution_engine.models.evidence import Evidence, SourceRef
from constitution_engine.models.orientation import Orientation
from constitution_engine.models.types import Confidence, new_id, now_utc

from constitution_providers.context import EpisodeContext
from constitution_providers.llm.dispatch import dispatch
from constitution_providers.llm.packing import default_reasoner_pack_v1
from constitution_providers.llm.registry import AdapterRegistry, ModelRouteSpec, PackRegistry
from constitution_providers.llm.openai import OpenAIAdapter, OpenAIClientStub


def _fixture_json(*, ev_id: str) -> Mapping[str, Any]:
    # Minimal provider_rules-compatible JSON payload
    return {
        "interpretations": [
            {
                "interpretation_id": "prov_int_1",
                "info_type": "hypothesis",
                "text": "We should gather more info before committing.",
                "confidence": 0.6,
                "uncertainty": {"level": 0.7},
                "evidence_refs": [ev_id],
                "limits": "test-limits",
                "meta": {"mode": "test"},
            }
        ],
        "options": [
            {
                "option_id": "prov_opt_1",
                "kind": "info_gathering",
                "title": "Ask clarifying questions",
                "description": "Collect missing inputs before selecting a commit action.",
                "action_class": "probe",
                "impact": 0.2,
                "reversibility": 0.9,
                "confidence": 0.6,
                "uncertainty": {"level": 0.7},
                "evidence_refs": [ev_id],
                "limits": "test-limits",
                "meta": {"mode": "test"},
            }
        ],
        "ranked_options": [
            {
                "rank": 1,
                "option_ref": "prov_opt_1",
                "rationale": "Safe first move under uncertainty.",
                "title": "Ask clarifying questions",
                "confidence": 0.6,
                "uncertainty": {"level": 0.7},
                "evidence_refs": [ev_id],
                "limits": "test-limits",
                "meta": {"mode": "test"},
            }
        ],
        "override_suggestions": [],
    }


def test_llm_dispatch_smoke() -> None:
    # --- Arrange: minimal EpisodeContext + one evidence item that proposals can reference ---
    ori = Orientation(meta={"mode": "test"})
    ev = Evidence(
        evidence_id=new_id("ev"),
        created_at=now_utc(),
        sources=(SourceRef(uri="stub://source", extra={"mode": "test"}),),
        spans=tuple(),
        summary="Stub evidence summary.",
        notes={"mode": "test"},
        integrity=Confidence(1.0),
    )
    ctx = EpisodeContext(orientation=ori, raw_inputs=tuple(), evidence=(ev,), meta={"mode": "test"})

    # Local registries (avoid relying on global defaults)
    packs = PackRegistry()
    packs.register(default_reasoner_pack_v1())

    adapters = AdapterRegistry()
    fixture = _fixture_json(ev_id=ev.evidence_id)
    adapters.register(
        "openai_stub",
        lambda: OpenAIAdapter(client=OpenAIClientStub(fixture_json=fixture)),
    )

    route = ModelRouteSpec(
        model_id="gpt-test",
        provider_id="openai_reasoner",
        adapter_key="openai_stub",
        pack_id="reasoner",
        temperature=0.0,
        limits="test-limits",
        provider_version="v1",
        extra_meta={"mode": "test"},
    )

    # --- Act ---
    ps = dispatch(ctx=ctx, route=route, packs=packs, adapters=adapters)

    # --- Assert: basic ProposalSet wiring ---
    assert ps.provider_id == "openai_reasoner"
    assert ps.model_id == "gpt-test"
    assert ps.limits == "test-limits"
    assert isinstance(getattr(ps, "run_id", None), str) and bool(ps.run_id.strip())
    assert getattr(getattr(ps, "sampling", None), "temperature", None) == 0.0

    # --- Assert: provider boundary rules pass (no violations) ---
    evidence_by_id = {ev.evidence_id: ev}
    violations = validate_proposalset(ps, evidence_by_id=evidence_by_id)
    assert violations == tuple()
