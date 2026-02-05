from constitution_engine.intake.adapter import draft_episode
from constitution_engine.intake.stub_drafter import StubDrafter
from constitution_engine.intake.types import GoalSpec, RawInputItem
from constitution_engine.intake.materialize import materialize_draft_episode

from constitution_engine.runtime.store import ArtifactStore
from constitution_engine.invariants.validate import validate_episode


def test_materialize_and_validate_smoke():
    store = ArtifactStore()

    goal = GoalSpec(goal_id="g1", statement="Decide whether to build a demo this month.", horizon_days=30)
    raw_inputs = [
        RawInputItem(
            raw_id="r1",
            text="I think the engine is solid now but I'm not sure people will understand it.",
            source_uri="internal:user",
            created_at_utc="2026-02-04T00:00:00Z",
        )
    ]

    draft = draft_episode(goal=goal, raw_inputs=raw_inputs, drafter=StubDrafter())
    episode_id = materialize_draft_episode(store=store, draft=draft)

    report = validate_episode(store=store, episode_id=episode_id)
    assert report.ok
    # (optional, if you want it explicit)
    assert report.violations == ()
    assert report.resolve_errors == ()
