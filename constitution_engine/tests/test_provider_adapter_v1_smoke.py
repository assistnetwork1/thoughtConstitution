# constitution_engine/tests/test_provider_adapter_v1_smoke.py
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from constitution_engine.intake.provider_adapter_v1 import draft_episode_from_proposals
from constitution_engine.intake.types import GoalSpec, RawInputItem
from constitution_engine.models.types import new_id
from constitution_providers.protocol.proposals import ProposalSet


def _make(cls: type[Any], **kwargs: Any) -> Any:
    """
    Defensive dataclass constructor: filters unknown kwargs so this test
    survives model field renames.
    """
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)
    return cls(**kwargs)


def test_provider_adapter_v1_smoke_protocol_to_draft_episode() -> None:
    # --- Arrange: minimal goal + local evidence creation path (via raw_inputs) ---
    goal = _make(
        GoalSpec,
        goal_id=new_id("goal"),
        statement="Decide what to do next, safely.",
        horizon_days=7,
        constraints=(),
        success_criteria=(),
    )

    raw_inputs = [
        _make(
            RawInputItem,
            raw_id="ri_1",
            text="We can bring dogs to offices to reduce stress.",
            source_uri="raw://user",
            created_at_utc="2026-02-05T00:00:00Z",
        )
    ]

    # Minimal provider output (even empty is allowed; adapter should not crash)
    ps = ProposalSet(provider_id="stub_reasoner", notes="empty proposals OK")

    # --- Act ---
    ep = draft_episode_from_proposals(
        goal=goal,
        raw_inputs=raw_inputs,
        proposal_sets=(ps,),
    )

    # --- Assert ---
    assert ep.episode_id
    assert ep.goal is not None
    assert len(ep.evidence) == 1  # evidence created from raw_inputs
    assert ep.options is not None
