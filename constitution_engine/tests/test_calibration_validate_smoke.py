from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.calibration import CalibrationNote
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.outcome import Outcome
from constitution_engine.models.recommendation import RankedOption, Recommendation
from constitution_engine.models.review import ReviewRecord
from constitution_engine.models.types import Confidence, Impact, Reversibility
from constitution_engine.runtime.store import ArtifactStore


def test_validate_episode_accepts_calibration_note_references():
    store = ArtifactStore()

    # Build an option that exists in-store
    opt = Option(
        option_id="opt_1",
        kind=OptionKind.EXECUTE,
        title="Do the thing",
        description="Execute the chosen action.",
        impact=Impact(0.2),
        reversibility=Reversibility(0.8),
        uncertainties=tuple(),
        action_class="probe",
    )

    # RankedOption carries confidence (Recommendation does not in your model)
    ro = RankedOption(
        option_id="opt_1",
        rank=1,
        score=0.8,
        rationale="Low risk; try it.",
        confidence=Confidence(0.6),
        uncertainties=tuple(),
    )

    # Build a recommendation that ranks that option
    rec = Recommendation(
        recommendation_id="rec_1",
        orientation_id="ori_1",
        ranked_options=(ro,),
    )

    # Episode indexes everything by IDs
    ep = (
        DecisionEpisode()
        .add_options("opt_1")
        .add_recommendations("rec_1")
        .add_reviews("rev_1")
        .add_outcomes("out_1")
    )

    out = Outcome(
        outcome_id="out_1",
        recommendation_id="rec_1",
        chosen_option_id="opt_1",
        description="Observed: completed successfully.",
    )

    # Minimal: must have review_id and episode_id in most designs.
    rev = ReviewRecord(review_id="rev_1", episode_id=ep.episode_id)

    store.put(ep)
    store.put(opt)
    store.put(rec)
    store.put(out)
    store.put(rev)

    cal = CalibrationNote(
        episode_id=ep.episode_id,
        review_id="rev_1",
        outcome_ids=("out_1",),
        summary="Next time require more evidence.",
        proposed_changes=("Require 2 sources before COMMIT",),
    )
    store.put(cal)

    ep2 = ep.add_calibrations(cal.calibration_id)
    store.put(ep2)

    report = validate_episode(store, ep2.episode_id)
    assert report.ok
