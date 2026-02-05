"""
This test locks the intake adapter contract.

If this test fails, DO NOT weaken models to fix it.
Either:
- Fix the adapter, or
- Create a new adapter version.
"""


from __future__ import annotations

from constitution_engine.intake.adapter import draft_episode
from constitution_engine.intake.stub_drafter import StubDrafter
from constitution_engine.intake.types import GoalSpec, RawInputItem
from constitution_engine.models.option import OptionKind


def test_draft_episode_smoke():
    goal = GoalSpec(goal_id="g1", statement="Decide whether to build a demo this month.", horizon_days=30)

    raw_inputs = [
        RawInputItem(
            raw_id="r1",
            text="I think the engine is solid now but I'm not sure people will understand it.",
            source_uri="internal:user",
            created_at_utc="2026-02-04T00:00:00Z",
        )
    ]

    ep = draft_episode(goal=goal, raw_inputs=raw_inputs, drafter=StubDrafter())

    assert ep.evidence
    assert ep.observations
    assert ep.options
    assert ep.recommendation is not None
    assert any(o.kind == OptionKind.INFO_GATHERING for o in ep.options)
