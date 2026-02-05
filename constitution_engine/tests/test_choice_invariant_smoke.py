# tests/test_choice_invariant_smoke.py
from __future__ import annotations

from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.choice import ChoiceRecord
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import RankedOption, Recommendation
from constitution_engine.models.types import Impact, Reversibility, Uncertainty, new_id, now_utc
from constitution_engine.runtime.store import ArtifactStore


def _make_execute_option(*, option_id: str, orientation_id: str) -> Option:
    return Option(
        option_id=option_id,
        kind=OptionKind.EXECUTE,
        title="Do the thing",
        description="Execute option for smoke test",
        orientation_id=orientation_id,
        impact=Impact(value=0.2),
        reversibility=Reversibility(value=0.9),
        uncertainties=(Uncertainty(level=0.2, description="low uncertainty"),),
        action_class="probe",  # safest valid class for any risk bucket
        observation_ids=tuple(),
        interpretation_ids=tuple(),
        evidence_ids=tuple(),
    )


def _make_recommendation(*, rec_id: str, orientation_id: str, option_id: str) -> Recommendation:
    return Recommendation(
        recommendation_id=rec_id,
        orientation_id=orientation_id,
        ranked_options=(
            RankedOption(
                option_id=option_id,
                rank=1,
                score=0.9,
                rationale="Top pick",
            ),
        ),
        evidence_ids=("ev_stub",),  # non-empty provenance pointer required by invariants
        observation_ids=tuple(),
        interpretation_ids=tuple(),
        model_state_ids=tuple(),
        override_used=False,
        override_scope_used=tuple(),
    )


def test_episode_acted_requires_choice_id_trips_invariant():
    store = ArtifactStore()

    orientation_id = new_id("or")
    option_id = new_id("op")
    rec_id = new_id("rc")

    opt = _make_execute_option(option_id=option_id, orientation_id=orientation_id)
    rec = _make_recommendation(rec_id=rec_id, orientation_id=orientation_id, option_id=option_id)

    store.put(opt)
    store.put(rec)

    # Acted=True but no choice_ids => should trip INV-CHO-001
    ep = DecisionEpisode(
        episode_id=new_id("ep"),
        option_ids=(option_id,),
        recommendation_ids=(rec_id,),
        acted=True,
        chosen_option_id=option_id,
        acted_at=now_utc(),
        choice_ids=tuple(),
    )
    store.put(ep)

    report = validate_episode(store, ep.episode_id)
    rules = {v.rule for v in report.violations}
    assert "INV-CHO-001" in rules


def test_choice_must_reference_existing_rec_and_option_trips_invariant():
    store = ArtifactStore()

    orientation_id = new_id("or")
    option_id = new_id("op")
    rec_id = new_id("rc")

    opt = _make_execute_option(option_id=option_id, orientation_id=orientation_id)
    rec = _make_recommendation(rec_id=rec_id, orientation_id=orientation_id, option_id=option_id)

    store.put(opt)
    store.put(rec)

    missing_rec_id = new_id("rc")
    missing_opt_id = new_id("op")

    ch = ChoiceRecord(
        choice_id=new_id("ch"),
        episode_id=new_id("ep"),
        recommendation_id=missing_rec_id,
        option_id=missing_opt_id,
        created_at=now_utc(),
        used_override=False,
        rationale="",
    )
    store.put(ch)

    ep = DecisionEpisode(
        episode_id=ch.episode_id,
        option_ids=(option_id,),
        recommendation_ids=(rec_id,),
        choice_ids=(ch.choice_id,),
        acted=True,
        chosen_option_id=option_id,
        acted_at=now_utc(),
    )
    store.put(ep)

    report = validate_episode(store, ep.episode_id)
    rules = {v.rule for v in report.violations}
    assert "INV-CHO-002" in rules


def test_choice_option_must_be_ranked_unless_override_trips_invariant():
    store = ArtifactStore()

    orientation_id = new_id("or")
    option_ranked = new_id("op")
    option_unranked = new_id("op")
    rec_id = new_id("rc")

    opt1 = _make_execute_option(option_id=option_ranked, orientation_id=orientation_id)
    opt2 = _make_execute_option(option_id=option_unranked, orientation_id=orientation_id)
    rec = _make_recommendation(rec_id=rec_id, orientation_id=orientation_id, option_id=option_ranked)

    store.put(opt1)
    store.put(opt2)
    store.put(rec)

    ch = ChoiceRecord(
        choice_id=new_id("ch"),
        episode_id=new_id("ep"),
        recommendation_id=rec_id,
        option_id=option_unranked,  # not ranked by rec
        created_at=now_utc(),
        used_override=False,
        rationale="",
    )
    store.put(ch)

    ep = DecisionEpisode(
        episode_id=ch.episode_id,
        option_ids=(option_ranked, option_unranked),
        recommendation_ids=(rec_id,),
        choice_ids=(ch.choice_id,),
        acted=True,
        chosen_option_id=option_unranked,
        acted_at=now_utc(),
    )
    store.put(ep)

    report = validate_episode(store, ep.episode_id)
    rules = {v.rule for v in report.violations}
    assert "INV-CHO-003" in rules


def test_choice_option_unranked_allowed_when_override_flag_set():
    store = ArtifactStore()

    orientation_id = new_id("or")
    option_ranked = new_id("op")
    option_unranked = new_id("op")
    rec_id = new_id("rc")

    opt1 = _make_execute_option(option_id=option_ranked, orientation_id=orientation_id)
    opt2 = _make_execute_option(option_id=option_unranked, orientation_id=orientation_id)
    rec = _make_recommendation(rec_id=rec_id, orientation_id=orientation_id, option_id=option_ranked)

    store.put(opt1)
    store.put(opt2)
    store.put(rec)

    ch = ChoiceRecord(
        choice_id=new_id("ch"),
        episode_id=new_id("ep"),
        recommendation_id=rec_id,
        option_id=option_unranked,  # not ranked
        created_at=now_utc(),
        used_override=True,  # allowed
        rationale="Emergency exception",
    )
    store.put(ch)

    ep = DecisionEpisode(
        episode_id=ch.episode_id,
        option_ids=(option_ranked, option_unranked),
        recommendation_ids=(rec_id,),
        choice_ids=(ch.choice_id,),
        acted=True,
        chosen_option_id=option_unranked,
        acted_at=now_utc(),
    )
    store.put(ep)

    report = validate_episode(store, ep.episode_id)
    rules = {v.rule for v in report.violations}
    assert "INV-CHO-003" not in rules
