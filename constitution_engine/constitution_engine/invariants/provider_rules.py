# constitution_engine/constitution_engine/invariants/provider_rules.py
"""
Provider-boundary invariants (v0.1) — ProposalSet validation.

Purpose:
- Treat provider outputs (LLMs / symbolic reasoners) as untrusted
- Enforce a hard constitutional boundary before any canonicalization
- Deterministic, test-lockable, no heuristics

Design:
- Accepts ProposalSet as either a dict-like Mapping or a dataclass-like object.
- Returns Sequence[InvariantViolation] using the canonical InvariantViolation type.

Hard constraints:
- No agent may emit kernel-owned artifacts (Recommendation/Choice/Outcome/Review/Calibration/Override).
- Action-relevant provider artifacts must declare confidence, uncertainty, provenance pointers, and limits.
- Evidence references must resolve against known Evidence IDs.
- RankedOptions must reference in-set options and form a strict total order.
- Override suggestions must be non-executable.

Public API:
- validate_proposalset(ps, evidence_by_id=...) -> Sequence[InvariantViolation]
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from constitution_engine.invariants.rules import InvariantViolation

# ---------------------------
# Helpers (dict OR dataclass)
# ---------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


# ---------------------------
# INV-PS-001 — ProposalSet schema admissible
# ---------------------------


def require_proposalset_header_fields(ps: Any) -> Sequence[InvariantViolation]:
    missing: list[str] = []

    if not _is_nonempty_str(_get(ps, "provider_id")):
        missing.append("provider_id")
    if not _is_nonempty_str(_get(ps, "model_id")):
        missing.append("model_id")
    if not _is_nonempty_str(_get(ps, "run_id")):
        missing.append("run_id")
    if not _is_nonempty_str(_get(ps, "limits")):
        missing.append("limits")

    sampling = _get(ps, "sampling", None)
    temp = _get(sampling, "temperature", None)
    if not isinstance(temp, (int, float)):
        missing.append("sampling.temperature")

    if missing:
        return (
            InvariantViolation(
                rule="INV-PS-001",
                message=f"ProposalSet is missing required header fields: {', '.join(missing)}",
            ),
        )
    return tuple()


# ---------------------------
# INV-PA-001 — Forbidden artifact classes are not present
# ---------------------------

_FORBIDDEN_FIELDS = {
    "recommendation",
    "choice_record",
    "outcome",
    "review_record",
    "calibration_note",
    "override",  # real override artifact
}


def require_no_forbidden_artifact_fields(ps: Any) -> Sequence[InvariantViolation]:
    for f in _FORBIDDEN_FIELDS:
        val = _get(ps, f, None)
        if val is not None:
            return (
                InvariantViolation(
                    rule="INV-PA-001",
                    message=f"ProposalSet contains forbidden artifact type: {f}",
                ),
            )
    return tuple()


# ---------------------------
# INV-PA-002 — Action-relevant provider artifacts must declare fields
# ---------------------------


def _require_required_fields(kind: str, obj: Any, obj_id: str) -> list[InvariantViolation]:
    missing: list[str] = []

    conf = _get(obj, "confidence", None)
    if not isinstance(conf, (int, float)):
        missing.append("confidence")

    unc = _get(obj, "uncertainty", None)
    level = _get(unc, "level", None)
    if not isinstance(level, (int, float)):
        missing.append("uncertainty.level")

    ev_refs = _get(obj, "evidence_refs", None)
    if not isinstance(ev_refs, (list, tuple)):
        missing.append("evidence_refs")

    lim = _get(obj, "limits", None)
    if not _is_nonempty_str(lim):
        missing.append("limits")

    if missing:
        return [
            InvariantViolation(
                rule="INV-PA-002",
                message=f"{kind} {obj_id} missing required fields: {', '.join(missing)}",
            )
        ]
    return []


def require_provider_artifacts_declare_fields(ps: Any) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []

    for it in _as_list(_get(ps, "interpretations", [])):
        iid = _get(it, "interpretation_id", "<?>")
        violations.extend(_require_required_fields("Interpretation", it, str(iid)))

    for opt in _as_list(_get(ps, "options", [])):
        oid = _get(opt, "option_id", "<?>")
        violations.extend(_require_required_fields("Option", opt, str(oid)))

    for ro in _as_list(_get(ps, "ranked_options", [])):
        rid = f"rank={_get(ro,'rank','<?>')}, option_ref={_get(ro,'option_ref','<?>')}"
        violations.extend(_require_required_fields("RankedOption", ro, rid))

    for osug in _as_list(_get(ps, "override_suggestions", [])):
        sid = f"invariant={_get(osug,'invariant_id','<?>')}"
        violations.extend(_require_required_fields("OverrideSuggestion", osug, sid))

    return tuple(violations)


# ---------------------------
# INV-PS-002 — Evidence references must resolve
# ---------------------------


def require_proposalset_evidence_refs_resolve(
    ps: Any,
    *,
    evidence_by_id: Mapping[str, Any],
) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []

    def check_refs(kind: str, obj: Any, label: str) -> None:
        refs = _get(obj, "evidence_refs", []) or []
        missing = [r for r in refs if r not in evidence_by_id]
        if missing:
            violations.append(
                InvariantViolation(
                    rule="INV-PS-002",
                    message=f"{kind} {label} references unknown evidence_id: {', '.join(missing)}",
                )
            )

    for it in _as_list(_get(ps, "interpretations", [])):
        check_refs("Interpretation", it, str(_get(it, "interpretation_id", "<?>")))

    for opt in _as_list(_get(ps, "options", [])):
        check_refs("Option", opt, str(_get(opt, "option_id", "<?>")))

    for ro in _as_list(_get(ps, "ranked_options", [])):
        label = f"rank={_get(ro,'rank','<?>')}, option_ref={_get(ro,'option_ref','<?>')}"
        check_refs("RankedOption", ro, label)

    for osug in _as_list(_get(ps, "override_suggestions", [])):
        check_refs("OverrideSuggestion", osug, str(_get(osug, "invariant_id", "<?>")))

    return tuple(violations)


# ---------------------------
# INV-PR-001 — RankedOptions must reference existing proposed Options
# ---------------------------


def require_ranked_options_reference_existing_options(ps: Any) -> Sequence[InvariantViolation]:
    opts = _as_list(_get(ps, "options", []))
    option_ids = {str(_get(o, "option_id", "")) for o in opts if _get(o, "option_id", None)}

    violations: list[InvariantViolation] = []
    for ro in _as_list(_get(ps, "ranked_options", [])):
        ref = _get(ro, "option_ref", None)
        if ref and str(ref) not in option_ids:
            violations.append(
                InvariantViolation(
                    rule="INV-PR-001",
                    message=f"RankedOption references missing option_id: {ref}",
                )
            )
    return tuple(violations)


# ---------------------------
# INV-PR-002 — RankedOptions must be a strict total order (no duplicates/gaps)
# ---------------------------


def require_ranked_options_strict_total_order(ps: Any) -> Sequence[InvariantViolation]:
    ros = _as_list(_get(ps, "ranked_options", []))
    if not ros:
        return tuple()

    ranks = [_get(r, "rank", None) for r in ros]
    if not all(isinstance(r, int) for r in ranks):
        return (
            InvariantViolation(
                rule="INV-PR-002",
                message="RankedOptions must be a strict total order (duplicates or gaps found)",
            ),
        )

    if len(set(ranks)) != len(ranks):
        return (
            InvariantViolation(
                rule="INV-PR-002",
                message="RankedOptions must be a strict total order (duplicates or gaps found)",
            ),
        )

    expected = list(range(1, len(ranks) + 1))
    if sorted(ranks) != expected:
        return (
            InvariantViolation(
                rule="INV-PR-002",
                message="RankedOptions must be a strict total order (duplicates or gaps found)",
            ),
        )

    refs = [_get(r, "option_ref", None) for r in ros]
    if len(set(refs)) != len(refs):
        return (
            InvariantViolation(
                rule="INV-PR-002",
                message="RankedOptions must be a strict total order (duplicates or gaps found)",
            ),
        )

    return tuple()


# ---------------------------
# INV-PS-003 — Override suggestions must be non-executable
# ---------------------------

_EXEC_OVERRIDE_FIELDS = {"override_id", "apply_override", "approved_by", "expires_at"}


def require_override_suggestions_non_executable(ps: Any) -> Sequence[InvariantViolation]:
    violations: list[InvariantViolation] = []
    for osug in _as_list(_get(ps, "override_suggestions", [])):
        if isinstance(osug, Mapping):
            bad = [k for k in osug.keys() if k in _EXEC_OVERRIDE_FIELDS]
        else:
            bad = [k for k in _EXEC_OVERRIDE_FIELDS if getattr(osug, k, None) is not None]
        if bad:
            violations.append(
                InvariantViolation(
                    rule="INV-PS-003",
                    message="Override suggestion contains executable override fields",
                )
            )
    return tuple(violations)


# ---------------------------
# Public API
# ---------------------------


def validate_proposalset(
    ps: Any,
    *,
    evidence_by_id: Mapping[str, Any],
) -> Sequence[InvariantViolation]:
    """
    Validate a single ProposalSet against the provider-boundary invariant pack.

    Strict: This function does not decide accept/reject policy; it only returns violations.
    Callers may reject the whole ProposalSet if any violations exist.
    """
    violations: list[InvariantViolation] = []
    violations.extend(require_proposalset_header_fields(ps))
    violations.extend(require_no_forbidden_artifact_fields(ps))
    violations.extend(require_provider_artifacts_declare_fields(ps))
    violations.extend(require_proposalset_evidence_refs_resolve(ps, evidence_by_id=evidence_by_id))
    violations.extend(require_ranked_options_reference_existing_options(ps))
    violations.extend(require_ranked_options_strict_total_order(ps))
    violations.extend(require_override_suggestions_non_executable(ps))
    return tuple(violations)
