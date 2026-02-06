# constitution_providers/llm/openai/tests/test_openai_adapter_contract.py
from __future__ import annotations

import json

from constitution_engine.invariants.provider_rules import validate_proposalset
from constitution_engine.models.evidence import Evidence, SourceRef
from constitution_engine.models.types import Confidence, new_id, now_utc
from constitution_engine.models.orientation import Orientation

from constitution_providers.context import EpisodeContext
from constitution_providers.llm.openai import OpenAIAdapter, OpenAIClientStub
from constitution_providers.llm.openai.packing import RenderedPrompt


def _fixture_json(*, ev_id: str) -> str:
    """
    Minimal JSON payload that should parse cleanly into a provider_rules-compatible ProposalSet.
    """
    payload = {
        "interpretations": [
            {
                "interpretation_id": "prov_int_1",
                "info_type": "hypothesis",
                "text": "We likely need more info before committing.",
                "confidence": 0.6,
                "uncertainty": {"level": 0.6},
                "evidence_refs": [ev_id],
                "limits": "stub: no external browsing; using provided context only.",
                "meta": {"source": "stub"},
            }
        ],
        "options": [
            {
                "option_id": "prov_opt_1",
                "kind": "info_gathering",
                "title": "Ask 3 clarifying questions",
                "description": "Probe: ask targeted questions to reduce uncertainty before deciding.",
                "action_class": "probe",
                "impact": 0.2,
                "reversibility": 0.95,
                "confidence": 0.7,
                "uncertainty": {"level": 0.4},
                "evidence_refs": [ev_id],
                "limits": "stub: no external browsing; using provided context only.",
                "meta": {"source": "stub"},
            },
            {
                "option_id": "prov_opt_2",
                "kind": "execute",
                "title": "Draft a lightweight plan",
                "description": "Limited: draft a plan assuming the current constraints, mark assumptions.",
                "action_class": "limited",
                "impact": 0.4,
                "reversibility": 0.85,
                "confidence": 0.55,
                "uncertainty": {"level": 0.55},
                "evidence_refs": [ev_id],
                "limits": "stub: no external browsing; using provided context only.",
                "meta": {"source": "stub"},
            },
        ],
        "ranked_options": [
            {
                "rank": 1,
                "option_ref": "prov_opt_1",
                "rationale": "Safest default: reduce uncertainty first.",
                "title": "Ask 3 clarifying questions",
                "confidence": 0.65,
                "uncertainty": {"level": 0.45},
                "evidence_refs": [ev_id],
                "limits": "stub: no external browsing; using provided context only.",
                "meta": {"source": "stub"},
            },
            {
                "rank": 2,
                "option_ref": "prov_opt_2",
                "rationale": "Proceed cautiously with explicit assumptions.",
                "title": "Draft a lightweight plan",
                "confidence": 0.55,
                "uncertainty": {"level": 0.55},
                "evidence_refs": [ev_id],
                "limits": "stub: no external browsing; using provided context only.",
                "meta": {"source": "stub"},
            },
        ],
        "override_suggestions": [],
    }
    return json.dumps(payload)


def test_openai_adapter_contract_parses_and_passes_provider_rules() -> None:
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

    fixture = _fixture_json(ev_id=ev.evidence_id)
    adapter = OpenAIAdapter(client=OpenAIClientStub(fixture_json=fixture))

    rendered = RenderedPrompt(
        pack_id="reasoner",
        pack_version="v1",
        system="SYSTEM",
        user="USER",
        meta={"pack_id": "reasoner", "pack_version": "v1"},
    )

    # --- Act: invoke + parse into ProposalSet ---
    payload = adapter.invoke(rendered_prompt=rendered, model_id="gpt-stub", temperature=0.0)

    ps = adapter.parse_to_proposalset(
        payload=payload,
        ctx=ctx,
        provider_id="openai_reasoner",
        model_id="gpt-stub",
        limits="stub: no external browsing; using provided context only.",
        temperature=0.0,
        provider_version="v1",
        prompt_meta=rendered.meta,
    )

    # --- Assert: provider-boundary invariants accept it ---
    violations = validate_proposalset(ps, evidence_by_id={ev.evidence_id: ev})
    assert violations == tuple(), f"Expected no violations, got: {violations}"

    # Extra sanity checks: ranking references in-set options + strict total order
    option_ids = {o.option_id for o in ps.options}
    assert all(ro.option_ref in option_ids for ro in ps.ranked_options)
    assert sorted(ro.rank for ro in ps.ranked_options) == list(range(1, len(ps.ranked_options) + 1))
