from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.calibration import CalibrationNote
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.outcome import Outcome
from constitution_engine.models.review import ReviewRecord
from constitution_engine.runtime.store import ArtifactStore


def test_calibration_episode_id_must_match_episode():
    store = ArtifactStore()

    ep = DecisionEpisode().add_reviews("rev_1").add_outcomes("out_1")
    store.put(ep)

    out = Outcome(outcome_id="out_1", description="Observed.")
    rev = ReviewRecord(review_id="rev_1", episode_id=ep.episode_id)
    store.put(out)
    store.put(rev)

    # Wrong episode_id on purpose
    cal = CalibrationNote(
        episode_id="ep_wrong",
        review_id="rev_1",
        outcome_ids=("out_1",),
        summary="Calibration tied to wrong episode.",
        proposed_changes=("Example",),
    )
    store.put(cal)

    ep2 = ep.add_calibrations(cal.calibration_id)
    store.put(ep2)

    report = validate_episode(store, ep2.episode_id)
    assert not report.ok
    assert any(v.rule == "INV-CAL-001" for v in report.violations)
