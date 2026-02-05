# tests/test_action_gating.py
#
# v0.5.1 ActionClass gating tests.
#
# IMPORTANT SEMANTICS:
# - evaluate_option_legality() returns:
#     (is_allowed, requires_override, reason)
# - requires_override == True means: "the option violates the gate and would need an override to be allowed"
#   This can be True even when the current Orientation does NOT permit overrides (ADVISORY_ONLY).
#
# This file locks:
# - Riskiness derivation (small, auditable lookup)
# - Key gate cases (high-risk/high-uncertainty => PROBE only; conservative removes COMMIT)
# - Override structural requirements (explicit, scoped, subset-checked)
# - End-to-end consistency: evaluate_option_legality agrees with allowed_action_classes

from __future__ import annotations

import pytest

from constitution_engine.invariants.spec_action_gating import (  # type: ignore
    ActionClass,
    GovernanceMode,
    Level3,
    Option,
    Orientation,
    RiskPosture,
    allowed_action_classes,
    derived_riskiness,
    impact_level,
    reversibility_level,
    uncertainty_level,
    evaluate_option_legality,
)

# Representative scalars for each band.
# If you ever change banding thresholds, adjust these to remain safely inside bands.
SCALAR_BY_LEVEL = {
    Level3.LOW: 0.10,   # safely < 1/3
    Level3.MED: 0.50,   # safely in [1/3, 2/3)
    Level3.HIGH: 0.90,  # safely >= 2/3
}

# --- Riskiness golden: adjust ONLY if you intentionally change the lookup rules.
RISKINESS_GOLDEN = {
    (Level3.LOW,  Level3.HIGH): Level3.LOW,
    (Level3.HIGH, Level3.LOW):  Level3.HIGH,
    # Everything else => MED under the "thin" lookup described in v0.5.1 patch notes
    (Level3.LOW,  Level3.MED):  Level3.MED,
    (Level3.LOW,  Level3.LOW):  Level3.MED,
    (Level3.MED,  Level3.HIGH): Level3.MED,
    (Level3.MED,  Level3.MED):  Level3.MED,
    (Level3.MED,  Level3.LOW):  Level3.MED,
    (Level3.HIGH, Level3.HIGH): Level3.MED,
    (Level3.HIGH, Level3.MED):  Level3.MED,
}


@pytest.mark.parametrize("imp_level", [Level3.LOW, Level3.MED, Level3.HIGH])
@pytest.mark.parametrize("rev_level", [Level3.LOW, Level3.MED, Level3.HIGH])
def test_derived_riskiness_golden(imp_level: Level3, rev_level: Level3) -> None:
    expected = RISKINESS_GOLDEN[(imp_level, rev_level)]
    got = derived_riskiness(imp_level, rev_level)
    assert got == expected


def test_gate_key_case_high_risk_high_uncertainty_probe_only() -> None:
    """
    Canonical: HIGH risk + HIGH uncertainty => PROBE only.
    We explicitly construct (impact=HIGH, reversibility=LOW) => riskiness=HIGH under golden rules.
    """
    risk = derived_riskiness(Level3.HIGH, Level3.LOW)
    assert risk == Level3.HIGH  # sanity (matches golden)

    allowed = allowed_action_classes(risk, Level3.HIGH, posture=RiskPosture.DEFAULT)
    assert allowed == {ActionClass.PROBE}


def test_gate_key_case_med_risk_med_uncertainty_allows_probe_and_limited_only() -> None:
    """
    Canonical: MED risk + MED uncertainty => {PROBE, LIMITED}.
    Under thin lookup, most combos map to MED risk; we pick (MED, MED) explicitly.
    """
    risk = derived_riskiness(Level3.MED, Level3.MED)
    assert risk == Level3.MED  # sanity

    allowed = allowed_action_classes(risk, Level3.MED, posture=RiskPosture.DEFAULT)
    assert allowed == {ActionClass.PROBE, ActionClass.LIMITED}


def test_gate_key_case_conservative_removes_commit_when_present() -> None:
    """
    Canonical: CONSERVATIVE posture may tighten by removing COMMIT but never loosen.

    We test a LOW-risk case where DEFAULT should be maximally permissive.
    Under the thin riskiness lookup, LOW risk is achieved by (impact=LOW, reversibility=HIGH).
    """
    risk = derived_riskiness(Level3.LOW, Level3.HIGH)
    assert risk == Level3.LOW  # sanity (matches golden)

    default_allowed = allowed_action_classes(risk, Level3.LOW, posture=RiskPosture.DEFAULT)
    conservative_allowed = allowed_action_classes(risk, Level3.LOW, posture=RiskPosture.CONSERVATIVE)

    assert ActionClass.PROBE in default_allowed
    assert ActionClass.LIMITED in default_allowed
    assert ActionClass.COMMIT in default_allowed, "LOW risk DEFAULT should allow COMMIT (unless you intentionally changed canon)"

    assert conservative_allowed.issubset(default_allowed)
    assert ActionClass.COMMIT not in conservative_allowed, "CONSERVERVATIVE should remove COMMIT where it would otherwise be allowed"


@pytest.mark.parametrize("imp_level", [Level3.LOW, Level3.MED, Level3.HIGH])
@pytest.mark.parametrize("rev_level", [Level3.LOW, Level3.MED, Level3.HIGH])
@pytest.mark.parametrize("unc_level", [Level3.LOW, Level3.MED, Level3.HIGH])
@pytest.mark.parametrize("action_class", [ActionClass.PROBE, ActionClass.LIMITED, ActionClass.COMMIT])
@pytest.mark.parametrize("posture", [RiskPosture.DEFAULT, RiskPosture.CONSERVATIVE])
def test_evaluate_option_legality_agrees_with_allowed_action_classes_in_advisory_only(
    imp_level: Level3,
    rev_level: Level3,
    unc_level: Level3,
    action_class: ActionClass,
    posture: RiskPosture,
) -> None:
    """
    End-to-end consistency check.

    In ADVISORY_ONLY:
      - allowed should match whether action_class ∈ allowed_action_classes(...)
      - requires_override should be True iff the gate is violated (i.e., not allowed)
    """
    impact = SCALAR_BY_LEVEL[imp_level]
    reversibility = SCALAR_BY_LEVEL[rev_level]
    uncertainty = SCALAR_BY_LEVEL[unc_level]

    opt = Option(
        impact=impact,
        reversibility=reversibility,
        uncertainty=uncertainty,
        action_class=action_class,
        dependencies=("obs:1", "interp:1"),
    )
    ori = Orientation(
        governance_mode=GovernanceMode.ADVISORY_ONLY,
        risk_posture=posture,
    )

    # Sanity: banding is stable
    assert impact_level(impact) == imp_level
    assert reversibility_level(reversibility) == rev_level
    assert uncertainty_level(uncertainty) == unc_level

    risk = derived_riskiness(imp_level, rev_level)
    allowed_set = allowed_action_classes(risk, unc_level, posture=posture)
    expected_allowed = action_class in allowed_set
    expected_requires_override = not expected_allowed

    allowed, requires_override, reason = evaluate_option_legality(opt, ori)

    assert allowed is expected_allowed, (
        f"allowed mismatch: risk={risk} unc={unc_level} posture={posture} allowed_set={allowed_set}; {reason}"
    )
    assert requires_override is expected_requires_override, (
        f"requires_override mismatch: expected {expected_requires_override}; {reason}"
    )


def test_override_disallowed_without_permission() -> None:
    """
    Choose a disallowed case and ensure it is disallowed in ADVISORY_ONLY,
    with requires_override=True (gate violated).

    This also locks the semantics:
      requires_override == True means gate violation, not permission status.
    """
    opt = Option(
        impact=SCALAR_BY_LEVEL[Level3.HIGH],
        reversibility=SCALAR_BY_LEVEL[Level3.LOW],
        uncertainty=SCALAR_BY_LEVEL[Level3.HIGH],
        action_class=ActionClass.COMMIT,
        dependencies=("obs:1", "interp:2"),
    )
    ori = Orientation(governance_mode=GovernanceMode.ADVISORY_ONLY, risk_posture=RiskPosture.DEFAULT)

    allowed, requires_override, _ = evaluate_option_legality(opt, ori)
    assert allowed is False
    assert requires_override is True


def test_override_missing_fields_is_invalid() -> None:
    """
    EXTENDED_ALLOWED without required override fields must fail.
    """
    opt = Option(
        impact=SCALAR_BY_LEVEL[Level3.HIGH],
        reversibility=SCALAR_BY_LEVEL[Level3.LOW],
        uncertainty=SCALAR_BY_LEVEL[Level3.HIGH],
        action_class=ActionClass.COMMIT,
        dependencies=("obs:1",),
    )

    ori_missing_scope = Orientation(
        governance_mode=GovernanceMode.EXTENDED_ALLOWED,
        risk_posture=RiskPosture.DEFAULT,
        override_scope=None,
        override_rationale="Emergency",
    )
    allowed, requires_override, _ = evaluate_option_legality(opt, ori_missing_scope, override_scope_used={"ANY"})
    assert allowed is False
    assert requires_override is True

    ori_missing_rationale = Orientation(
        governance_mode=GovernanceMode.EXTENDED_ALLOWED,
        risk_posture=RiskPosture.DEFAULT,
        override_scope={"ALLOW_GATE_BYPASS"},
        override_rationale="   ",
    )
    allowed, requires_override, _ = evaluate_option_legality(opt, ori_missing_rationale, override_scope_used={"ALLOW_GATE_BYPASS"})
    assert allowed is False
    assert requires_override is True


def test_override_subset_enforced_allows_when_valid() -> None:
    """
    Valid override must be:
      - EXTENDED_ALLOWED
      - override_scope present
      - override_rationale present
      - override_scope_used present and subset of override_scope
    """
    opt = Option(
        impact=SCALAR_BY_LEVEL[Level3.HIGH],
        reversibility=SCALAR_BY_LEVEL[Level3.LOW],
        uncertainty=SCALAR_BY_LEVEL[Level3.HIGH],
        action_class=ActionClass.COMMIT,
        dependencies=("obs:1", "interp:2"),
    )

    ori = Orientation(
        governance_mode=GovernanceMode.EXTENDED_ALLOWED,
        risk_posture=RiskPosture.DEFAULT,
        override_scope={"ALLOW_GATE_BYPASS", "ALLOW_COMMIT_UNDER_HIGH_UNC"},
        override_rationale="Time-critical safety action; delay increases harm.",
    )

    allowed, requires_override, reason = evaluate_option_legality(
        opt,
        ori,
        override_scope_used={"ALLOW_GATE_BYPASS"},
    )
    assert allowed is True, reason
    assert requires_override is True

    allowed, requires_override, _ = evaluate_option_legality(
        opt,
        ori,
        override_scope_used={"NOT_IN_SCOPE"},
    )
    assert allowed is False
    assert requires_override is True


def test_dependencies_required_for_auditability() -> None:
    """
    Option.dependencies must be non-empty.
    """
    opt = Option(
        impact=0.1,
        reversibility=0.9,
        uncertainty=0.1,
        action_class=ActionClass.PROBE,
        dependencies=(),
    )
    ori = Orientation(governance_mode=GovernanceMode.ADVISORY_ONLY, risk_posture=RiskPosture.DEFAULT)

    allowed, requires_override, _ = evaluate_option_legality(opt, ori)
    assert allowed is False
    assert requires_override is False  # structural invalidity, not a gate violation
