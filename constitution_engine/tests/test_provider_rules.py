# tests/test_provider_rules.py
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from constitution_engine.invariants.provider_rules import validate_proposalset


def _good_evidence_by_id() -> Dict[str, Any]:
    # provider_rules only needs membership checks on keys
    return {"ev_1": object()}


def _good_proposalset() -> Dict[str, Any]:
    """
    Minimal ProposalSet dict that should PASS all provider invariants.

    Assumptions (per provider_rules v0.1):
    - Header requires: provider_id, model_id, run_id, limits, sampling.temperature
    - Artifacts require: confidence, uncertainty.level, evidence_refs, limits
    - evidence_refs must resolve via evidence_by_id keys
    - ranked_options must reference option_id in options and be strict total order starting at 1
    """
    return {
        "provider_id": "stub",
        "model_id": "stub-v0",
        "run_id": "run-1",
        "sampling": {"temperature": 0.0},
        "limits": "Stub provider; may miss edge cases.",
        "evidence_threads": [],
        "interpretations": [
            {
                "interpretation_id": "prov_tmp_int_1",
                "info_type": "hypothesis",
                "text": "A plausible hypothesis.",
                "confidence": 0.6,
                "uncertainty": {"level": 0.4},
                "evidence_refs": ["ev_1"],
                "limits": "May be wrong if context is incomplete.",
            }
        ],
        "options": [
            {
                "option_id": "prov_tmp_opt_1",
                "kind": "info_gathering",
                "title": "Ask clarifying questions",
                "description": "Gather missing inputs before acting.",
                "action_class": "probe",
                "impact": 0.2,
                "reversibility": 0.95,
                "confidence": 0.7,
                "uncertainty": {"level": 0.3},
                "evidence_refs": ["ev_1"],
                "limits": "Only helps if the other party responds.",
            }
        ],
        "ranked_options": [
            {
                "rank": 1,
                "option_ref": "prov_tmp_opt_1",
                "rationale": "Safest first move under uncertainty.",
                "confidence": 0.6,
                "uncertainty": {"level": 0.4},
                "evidence_refs": ["ev_1"],
                "limits": "Ranking may change if new evidence arrives.",
            }
        ],
        "override_suggestions": [],
    }


def _rules(violations: List[Any]) -> List[str]:
    return [v.rule for v in violations]


def test_valid_proposalset_passes() -> None:
    ps = _good_proposalset()
    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert violations == []


def test_missing_header_fields_trips_inv_ps_001() -> None:
    ps = _good_proposalset()
    ps.pop("provider_id", None)

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PS-001" in _rules(violations)


def test_forbidden_artifact_type_trips_inv_pa_001() -> None:
    ps = _good_proposalset()
    ps["recommendation"] = {"oops": True}  # forbidden

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PA-001" in _rules(violations)


def test_missing_required_fields_on_option_trips_inv_pa_002() -> None:
    ps = _good_proposalset()
    # remove required fields from the option
    ps["options"][0].pop("confidence", None)
    ps["options"][0].pop("uncertainty", None)

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PA-002" in _rules(violations)


def test_unknown_evidence_ref_trips_inv_ps_002() -> None:
    ps = _good_proposalset()
    # inject missing evidence id
    ps["options"][0]["evidence_refs"] = ["ev_missing"]

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PS-002" in _rules(violations)


def test_ranked_option_missing_option_ref_trips_inv_pr_001() -> None:
    ps = _good_proposalset()
    ps["ranked_options"][0]["option_ref"] = "prov_tmp_opt_DOES_NOT_EXIST"

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PR-001" in _rules(violations)


def test_ranked_options_strict_total_order_trips_inv_pr_002() -> None:
    ps = _good_proposalset()
    # add a second ranked option with a gap (rank 3 instead of 2)
    ps["options"].append(
        {
            "option_id": "prov_tmp_opt_2",
            "kind": "execute",
            "title": "Do something",
            "description": "Placeholder",
            "action_class": "limited",
            "impact": 0.4,
            "reversibility": 0.6,
            "confidence": 0.6,
            "uncertainty": {"level": 0.4},
            "evidence_refs": ["ev_1"],
            "limits": "Placeholder option.",
        }
    )
    ps["ranked_options"].append(
        {
            "rank": 3,  # gap => violates strict total order
            "option_ref": "prov_tmp_opt_2",
            "rationale": "Second choice.",
            "confidence": 0.6,
            "uncertainty": {"level": 0.4},
            "evidence_refs": ["ev_1"],
            "limits": "Placeholder ranking.",
        }
    )

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PR-002" in _rules(violations)


def test_override_suggestion_executable_fields_trips_inv_ps_003() -> None:
    ps = _good_proposalset()
    ps["override_suggestions"] = [
        {
            "invariant_id": "INV-ACT-002",
            "reason": "Test suggestion.",
            "scope": "episode_only",
            "confidence": 0.5,
            "uncertainty": {"level": 0.6},
            "evidence_refs": ["ev_1"],
            "limits": "Test only.",
            # executable field (forbidden)
            "apply_override": True,
        }
    ]

    violations = list(validate_proposalset(ps, evidence_by_id=_good_evidence_by_id()))
    assert "INV-PS-003" in _rules(violations)
