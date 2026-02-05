from constitution_engine.invariants.validate import validate_episode
from constitution_engine.intake.adapter import draft_episode
from constitution_engine.intake.materialize import materialize_draft_episode
from constitution_engine.intake.act import act_on_option
from constitution_engine.runtime.store import ArtifactStore

from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.testing.stubs import StubDrafter
from constitution_engine.intake.types import GoalSpec, RawInputItem


def test_acted_requires_outcome():
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
    ep_id = materialize_draft_episode(store=store, draft=draft)

    # Choose an option from the materialized episode (robust even if DraftEpisode.options is empty)
    ep = store.must_get(DecisionEpisode, ep_id)
    assert ep.option_ids, "materialize_draft_episode should store at least one option"
    chosen = ep.option_ids[0]

    # Act, but do NOT log an outcome yet => should violate INV-OUT-001
    act_on_option(store=store, episode_id=ep_id, chosen_option_id=chosen)

    report = validate_episode(store=store, episode_id=ep_id)
    assert any(v.rule == "INV-OUT-001" for v in report.violations)
