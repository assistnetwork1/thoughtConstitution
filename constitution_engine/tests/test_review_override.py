import pytest

from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.recommendation import Recommendation, RankedOption
from constitution_engine.models.review import ReviewRecord
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.runtime.store import ArtifactStore
from constitution_engine.models.types import Impact, Reversibility, Uncertainty
from constitution_engine.runtime.store import InMemoryArtifactStore



def _make_store() -> ArtifactStore:
    return ArtifactStore()


def test_inv_rev_001_override_used_requires_review_id() -> None:
    store = _make_store()

    # Option + Recommendation: override_used True (assumes your Recommendation has these fields)
    opt = Option(option_id="opt1").with_kind(OptionKind.EXECUTE).with_action_class("commit")
    rec = Recommendation(
        recommendation_id="rec1",
        orientation_id="ori1",
        ranked_options=(RankedOption(option_id="opt1", rank=1, score=0.0, rationale="x"),),
        evidence_ids=("ev1",),
        observation_ids=("obs1",),
        interpretation_ids=tuple(),
        model_state_ids=tuple(),
        override_used=True,
        override_scope_used=("ALLOW_GATE_BYPASS",),
    )

    ep = DecisionEpisode(
        episode_id="ep1",
        recommendation_ids=("rec1",),
        review_ids=tuple(),  # <-- missing
    )

    store.put(opt)
    store.put(rec)
    store.put(ep)

    report = validate_episode(store, "ep1")
    rules = {v.rule for v in report.violations}
    assert "INV-REV-001" in rules


def test_inv_rev_002_review_must_audit_overrides() -> None:
    store = _make_store()

    opt = Option(option_id="opt1").with_kind(OptionKind.EXECUTE).with_action_class("commit")
    rec = Recommendation(
        recommendation_id="rec1",
        orientation_id="ori1",
        ranked_options=(RankedOption(option_id="opt1", rank=1, score=0.0, rationale="x"),),
        evidence_ids=("ev1",),
        observation_ids=("obs1",),
        interpretation_ids=tuple(),
        model_state_ids=tuple(),
        override_used=True,
        override_scope_used=("ALLOW_GATE_BYPASS",),
    )

    # Review exists but missing audit entry
    rev = ReviewRecord(
        review_id="rev1",
        episode_id="ep1",
        override_audit={"overrides": []},  # <-- missing rec1 entry
    )

    ep = DecisionEpisode(
        episode_id="ep1",
        recommendation_ids=("rec1",),
        review_ids=("rev1",),
    )

    store.put(opt)
    store.put(rec)
    store.put(rev)
    store.put(ep)

    report = validate_episode(store, "ep1")
    rules = {v.rule for v in report.violations}
    assert "INV-REV-002" in rules


def test_override_review_valid_passes() -> None:
    store = _make_store()

    opt = Option(option_id="opt1").with_kind(OptionKind.EXECUTE).with_action_class("commit")
    rec = Recommendation(
        recommendation_id="rec1",
        orientation_id="ori1",
        ranked_options=(RankedOption(option_id="opt1", rank=1, score=0.0, rationale="x"),),
        evidence_ids=("ev1",),
        observation_ids=("obs1",),
        interpretation_ids=tuple(),
        model_state_ids=tuple(),
        override_used=True,
        override_scope_used=("ALLOW_GATE_BYPASS",),
    )

    rev = ReviewRecord(
        review_id="rev1",
        episode_id="ep1",
        override_audit={
            "overrides": [
                {
                    "recommendation_id": "rec1",
                    "override_scope_used": ["ALLOW_GATE_BYPASS"],
                    "rationale": "Time-critical action; probe not sufficient.",
                }
            ]
        },
    )

    ep = DecisionEpisode(
        episode_id="ep1",
        recommendation_ids=("rec1",),
        review_ids=("rev1",),
    )

    store.put(opt)
    store.put(rec)
    store.put(rev)
    store.put(ep)

    report = validate_episode(store, "ep1")
    assert report.ok, report.violations
def _make_store():
    return InMemoryArtifactStore()