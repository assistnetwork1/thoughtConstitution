from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Set, Tuple


# -----------------------------
# Enums (canonical governance)
# -----------------------------

class ActionClass(str, Enum):
    PROBE = "PROBE"
    LIMITED = "LIMITED"
    COMMIT = "COMMIT"


class GovernanceMode(str, Enum):
    ADVISORY_ONLY = "ADVISORY_ONLY"
    EXTENDED_ALLOWED = "EXTENDED_ALLOWED"


class Level3(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class RiskPosture(str, Enum):
    DEFAULT = "DEFAULT"
    CONSERVATIVE = "CONSERVATIVE"


# -----------------------------
# Minimal data shapes
# -----------------------------

@dataclass(frozen=True)
class Option:
    impact: float               # [0,1]
    reversibility: float        # [0,1] (1 = highly reversible, 0 = irreversible)
    uncertainty: float          # [0,1]
    action_class: ActionClass
    dependencies: Tuple[str, ...]  # artifact IDs


@dataclass(frozen=True)
class Orientation:
    governance_mode: GovernanceMode = GovernanceMode.ADVISORY_ONLY
    risk_posture: RiskPosture = RiskPosture.DEFAULT

    # Required if governance_mode == EXTENDED_ALLOWED
    override_scope: Optional[Set[str]] = None
    override_rationale: Optional[str] = None


# -----------------------------
# Scalar -> ordinal banding
# (stable "bands" for v0.5.1 bridge)
# -----------------------------

def band_scalar(x: float) -> Level3:
    """
    Simple 3-band mapping:
      [0.0, 0.333...) -> LOW
      [0.333..., 0.666...) -> MED
      [0.666..., 1.0] -> HIGH
    """
    if x < 1/3:
        return Level3.LOW
    if x < 2/3:
        return Level3.MED
    return Level3.HIGH


def impact_level(impact: float) -> Level3:
    return band_scalar(_clamp01(impact))


def uncertainty_level(unc: float) -> Level3:
    return band_scalar(_clamp01(unc))


def reversibility_level(rev: float) -> Level3:
    """
    Reversibility is inverted relative to risk:
      rev HIGH => safer
      rev LOW  => riskier
    We keep the ordinal label as "HIGH/MED/LOW" but interpret LOW as worse.
    """
    return band_scalar(_clamp01(rev))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# -----------------------------
# Riskiness derived from impact × reversibility
# (small, auditable lookup)
# -----------------------------

def derived_riskiness(imp: Level3, rev: Level3) -> Level3:
    """
    Intuition:
      - higher impact => higher riskiness
      - lower reversibility => higher riskiness

    Lookup-style rules:
      - (impact HIGH) and (reversibility LOW) => HIGH risk
      - (impact LOW)  and (reversibility HIGH) => LOW risk
      - otherwise => MED risk
    """
    if imp == Level3.HIGH and rev == Level3.LOW:
        return Level3.HIGH
    if imp == Level3.LOW and rev == Level3.HIGH:
        return Level3.LOW
    return Level3.MED


# -----------------------------
# Gate: allowed action classes
# -----------------------------

def allowed_action_classes(
    risk: Level3,
    unc: Level3,
    posture: RiskPosture = RiskPosture.DEFAULT,
) -> Set[ActionClass]:
    """
    Canonical gate (as in your doc):

      If risk=HIGH -> uncertainty must be LOW, else PROBE only
      If risk=MED  -> uncertainty must be <= MED, else PROBE only
      If risk=LOW  -> any uncertainty allowed (declared), baseline allows all

    Then optional posture tightening:
      CONSERVATIVE may remove COMMIT even when gate permits.
    """
    # Baseline allowance
    if risk == Level3.LOW:
        allowed = {ActionClass.PROBE, ActionClass.LIMITED, ActionClass.COMMIT}
    elif risk == Level3.MED:
        allowed = {ActionClass.PROBE} if unc == Level3.HIGH else {ActionClass.PROBE, ActionClass.LIMITED}
        # If you *want* MED+LOW uncertainty to allow COMMIT, add it explicitly here.
    else:  # risk == HIGH
        allowed = {ActionClass.PROBE} if unc != Level3.LOW else {ActionClass.PROBE, ActionClass.LIMITED}
        # If you *want* HIGH risk + LOW uncertainty to allow COMMIT, add it explicitly here.

    # Posture tightening (never loosens)
    if posture == RiskPosture.CONSERVATIVE and ActionClass.COMMIT in allowed:
        allowed = set(allowed)
        allowed.discard(ActionClass.COMMIT)

    return allowed


# -----------------------------
# Override validation
# -----------------------------

def override_is_valid(orientation: Orientation, scope_used: Optional[Set[str]]) -> Tuple[bool, str]:
    """
    Checks only the *structural* legitimacy of override availability.
    Scope semantics are intentionally simple: subset check.
    """
    if orientation.governance_mode != GovernanceMode.EXTENDED_ALLOWED:
        return False, "override not permitted: governance_mode != EXTENDED_ALLOWED"

    if not orientation.override_scope:
        return False, "override invalid: missing override_scope"

    if not orientation.override_rationale or not orientation.override_rationale.strip():
        return False, "override invalid: missing override_rationale"

    if not scope_used:
        return False, "override invalid: missing override_scope_used"

    if not scope_used.issubset(orientation.override_scope):
        return False, "override invalid: override_scope_used is not a subset of Orientation.override_scope"

    return True, "override valid"


# -----------------------------
# THE SPEC FUNCTION YOU ASKED FOR
# -----------------------------

def evaluate_option_legality(
    option: Option,
    orientation: Orientation,
    *,
    override_scope_used: Optional[Set[str]] = None,
) -> Tuple[bool, bool, str]:
    """
    Returns:
      (is_allowed, requires_override, reason_string)

    Interpretation:
      - If is_allowed=True and requires_override=False: gate satisfied normally.
      - If is_allowed=True and requires_override=True: gate violated but valid override exists.
      - If is_allowed=False: gate violated and override missing/invalid.

    Note: This function is purely constitutional. It does NOT rank options.
    """
    # Minimal structural checks that matter for governance
    if not option.dependencies:
        return False, False, "invalid option: dependencies required for auditability"

    imp = impact_level(option.impact)
    rev = reversibility_level(option.reversibility)
    unc = uncertainty_level(option.uncertainty)

    risk = derived_riskiness(imp, rev)
    allowed = allowed_action_classes(risk, unc, posture=orientation.risk_posture)

    if option.action_class in allowed:
        return True, False, f"allowed by gate: risk={risk.value}, uncertainty={unc.value}, allowed={sorted(a.value for a in allowed)}"

    # Gate violation → override path
    ok, msg = override_is_valid(orientation, override_scope_used)
    if ok:
        return True, True, f"allowed only by override: gate disallows {option.action_class.value}; {msg}"

    return False, True, f"disallowed: gate disallows {option.action_class.value} and {msg}"


# -----------------------------
# Example (quick sanity)
# -----------------------------
if __name__ == "__main__":
    opt = Option(
        impact=0.9,
        reversibility=0.1,
        uncertainty=0.8,
        action_class=ActionClass.COMMIT,
        dependencies=("obs:1", "interp:2"),
    )
    ori = Orientation(governance_mode=GovernanceMode.ADVISORY_ONLY, risk_posture=RiskPosture.DEFAULT)

    print(evaluate_option_legality(opt, ori))
