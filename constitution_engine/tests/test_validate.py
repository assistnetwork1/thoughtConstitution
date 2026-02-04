# tests/test_validate.py

from constitution_engine.invariants.validate import validate_all
from constitution_engine.models.option import OptionKind
from constitution_engine.models.types import Impact, Reversibility, Uncertainty


def test_recommendation_missing_ranked_options_trips_invariant(make_minimal_bundle):
    observations, evidence_items, options, rec = make_minimal_bundle(
        ranked_options=[]
    )

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "recommendation_ranked_options" in rules


def test_execute_option_missing_action_class_trips_v051_gate(make_minimal_bundle):
    """
    v0.5.1: ranked EXECUTE options must declare action_class (probe/limited/commit).
    This test ensures INV-ACT-001 is visible in validate_all.
    """
    observations, evidence_items, options, rec = make_minimal_bundle()

    assert rec.ranked_options, "fixture must produce at least one ranked option by default"
    top_id = rec.ranked_options[0].option_id

    new_options = []
    found = False
    for opt in options:
        if opt.option_id == top_id:
            found = True
            opt = opt.with_kind(OptionKind.EXECUTE)
            opt = opt.with_action_class(None)
        new_options.append(opt)

    assert found, f"top ranked option_id={top_id} not found in options bundle"

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=new_options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "INV-ACT-001" in rules


def test_v051_gate_allows_probe_for_risky_high_uncertainty_execute(make_minimal_bundle):
    """
    Positive gate test:
    - High impact + low reversibility => HIGH riskiness
    - High uncertainty => only PROBE allowed
    Therefore: action_class='probe' should NOT trip INV-ACT-002.
    """
    observations, evidence_items, options, rec = make_minimal_bundle()

    assert rec.ranked_options, "fixture must produce at least one ranked option by default"
    top_id = rec.ranked_options[0].option_id

    # impact=HIGH (0.9), reversibility=LOW (0.1), uncertainty=HIGH (0.9)
    risky_impact = Impact(0.9)
    low_rev = Reversibility(0.1)
    high_unc = Uncertainty("high uncertainty", level=0.9)

    new_options = []
    found = False
    for opt in options:
        if opt.option_id == top_id:
            found = True
            opt = opt.with_kind(OptionKind.EXECUTE)
            opt = opt.with_impact(risky_impact)
            opt = opt.with_reversibility(low_rev)
            opt = opt.add_uncertainties(high_unc)
            opt = opt.with_action_class("probe")
        new_options.append(opt)

    assert found, f"top ranked option_id={top_id} not found in options bundle"

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=new_options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "INV-ACT-001" not in rules
    assert "INV-ACT-002" not in rules


def test_v051_gate_blocks_commit_for_risky_high_uncertainty_execute(make_minimal_bundle):
    """
    Negative gate test:
    Same risk/uncertainty as the positive test, but action_class='commit'.
    For HIGH riskiness and uncertainty != LOW, allowed={PROBE} only,
    so COMMIT must trip INV-ACT-002.
    """
    observations, evidence_items, options, rec = make_minimal_bundle()

    assert rec.ranked_options, "fixture must produce at least one ranked option by default"
    top_id = rec.ranked_options[0].option_id

    risky_impact = Impact(0.9)
    low_rev = Reversibility(0.1)
    high_unc = Uncertainty("high uncertainty", level=0.9)

    new_options = []
    found = False
    for opt in options:
        if opt.option_id == top_id:
            found = True
            opt = opt.with_kind(OptionKind.EXECUTE)
            opt = opt.with_impact(risky_impact)
            opt = opt.with_reversibility(low_rev)
            opt = opt.add_uncertainties(high_unc)
            opt = opt.with_action_class("commit")
        new_options.append(opt)

    assert found, f"top ranked option_id={top_id} not found in options bundle"

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=new_options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "INV-ACT-001" not in rules
    assert "INV-ACT-002" in rules


def test_v051_gate_allows_limited_for_med_risk_med_uncertainty_execute(make_minimal_bundle):
    """
    Middle band positive:
      risk=MED and uncertainty=MED => allowed={PROBE, LIMITED}
    Therefore: action_class='limited' should NOT trip INV-ACT-002.
    """
    observations, evidence_items, options, rec = make_minimal_bundle()

    assert rec.ranked_options, "fixture must produce at least one ranked option by default"
    top_id = rec.ranked_options[0].option_id

    med_impact = Impact(0.5)
    med_rev = Reversibility(0.5)
    med_unc = Uncertainty("medium uncertainty", level=0.5)

    new_options = []
    found = False
    for opt in options:
        if opt.option_id == top_id:
            found = True
            opt = opt.with_kind(OptionKind.EXECUTE)
            opt = opt.with_impact(med_impact)
            opt = opt.with_reversibility(med_rev)
            opt = opt.add_uncertainties(med_unc)
            opt = opt.with_action_class("limited")
        new_options.append(opt)

    assert found, f"top ranked option_id={top_id} not found in options bundle"

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=new_options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "INV-ACT-001" not in rules
    assert "INV-ACT-002" not in rules


def test_v051_gate_blocks_commit_for_med_risk_med_uncertainty_execute(make_minimal_bundle):
    """
    Middle band negative:
      risk=MED and uncertainty=MED => allowed={PROBE, LIMITED}
    Therefore: action_class='commit' must be rejected (INV-ACT-002).
    """
    observations, evidence_items, options, rec = make_minimal_bundle()

    assert rec.ranked_options, "fixture must produce at least one ranked option by default"
    top_id = rec.ranked_options[0].option_id

    med_impact = Impact(0.5)
    med_rev = Reversibility(0.5)
    med_unc = Uncertainty("medium uncertainty", level=0.5)

    new_options = []
    found = False
    for opt in options:
        if opt.option_id == top_id:
            found = True
            opt = opt.with_kind(OptionKind.EXECUTE)
            opt = opt.with_impact(med_impact)
            opt = opt.with_reversibility(med_rev)
            opt = opt.add_uncertainties(med_unc)
            opt = opt.with_action_class("commit")
        new_options.append(opt)

    assert found, f"top ranked option_id={top_id} not found in options bundle"

    violations = validate_all(
        observations=observations,
        evidence_items=evidence_items,
        options=new_options,
        recommendation=rec,
    )

    rules = {v.rule for v in violations}
    assert "INV-ACT-001" not in rules
    assert "INV-ACT-002" in rules
