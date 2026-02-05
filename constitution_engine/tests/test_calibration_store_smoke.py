from constitution_engine.models.calibration import CalibrationNote
from constitution_engine.runtime.store import ArtifactStore

def test_store_put_get_calibration_note():
    store = ArtifactStore()
    cal = CalibrationNote(episode_id="ep_1", review_id="rev_1", summary="Adjust next time.")
    store.put(cal)
    got = store.must_get(CalibrationNote, cal.calibration_id)
    assert got.summary.startswith("Adjust")
