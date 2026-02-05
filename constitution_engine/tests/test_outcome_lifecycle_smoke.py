from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.outcome import Outcome
from constitution_engine.runtime.store import ArtifactStore


def test_episode_act_and_log_outcome_roundtrip():
    store = ArtifactStore()

    ep = DecisionEpisode().mark_acted(chosen_option_id="opt_123")
    assert ep.acted is True
    assert ep.chosen_option_id == "opt_123"
    assert ep.acted_at is not None

    out = Outcome(
        recommendation_id="rec_1",
        chosen_option_id="opt_123",
        description="Observed: completed successfully",
    )
    store.put(out)

    ep2 = ep.log_outcome(out.outcome_id)
    assert out.outcome_id in ep2.outcome_ids

    # store resolve sanity
    got = store.must_get(Outcome, out.outcome_id)
    assert got.description.startswith("Observed:")
