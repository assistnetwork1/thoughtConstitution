# constitution_engine/runtime/materialize.py
from __future__ import annotations

from typing import Any, Iterable, Tuple

from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.orientation import Orientation
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.raw_input import RawInput
from constitution_engine.runtime.store import ArtifactStore


def _persist_one(store: ArtifactStore, obj: Any) -> None:
    for method_name in ("add", "put", "upsert", "store", "insert"):
        m = getattr(store, method_name, None)
        if callable(m):
            m(obj)
            return
    raise AttributeError(
        "ArtifactStore has no supported persistence method. "
        "Expected one of: add/put/upsert/store/insert."
    )


def _persist_many(store: ArtifactStore, items: Iterable[Any]) -> None:
    for it in items:
        _persist_one(store, it)


def _normalize_violations(report: Any) -> Tuple[Any, ...]:
    """
    Your validator returns a ValidationReport (not directly iterable).
    Normalize to a stable tuple of violations.

    Canonical behavior for this repo:
    - If report has .violations and .resolve_errors and both are empty -> ()
    - If either has entries -> tuple(violations) + tuple(resolve_errors)
    - If report is already list/tuple -> tuple(report)
    - If report indicates ok/passed/is_ok -> ()
    - Otherwise -> (report,) as a last-resort “unknown failure shape”
    """
    if report is None:
        return tuple()

    # Some implementations return list/tuple directly
    if isinstance(report, tuple):
        return report
    if isinstance(report, list):
        return tuple(report)

    # ValidationReport-like object path
    v = getattr(report, "violations", None)
    if v is not None:
        re = getattr(report, "resolve_errors", None)

        # normalize violations
        if isinstance(v, tuple):
            v_tuple = v
        elif isinstance(v, list):
            v_tuple = tuple(v)
        else:
            try:
                v_tuple = tuple(v)  # type: ignore[arg-type]
            except TypeError:
                v_tuple = (v,)

        # normalize resolve_errors
        if re is None:
            re_tuple: Tuple[Any, ...] = tuple()
        elif isinstance(re, tuple):
            re_tuple = re
        elif isinstance(re, list):
            re_tuple = tuple(re)
        else:
            try:
                re_tuple = tuple(re)  # type: ignore[arg-type]
            except TypeError:
                re_tuple = (re,)

        # Clean report
        if not v_tuple and not re_tuple:
            return tuple()

        # Non-clean report: return both categories (stable)
        return tuple(v_tuple) + tuple(re_tuple)

    # If there is no .violations attribute, fall back to ok flags
    ok = getattr(report, "ok", None)
    passed = getattr(report, "passed", None)
    is_ok = getattr(report, "is_ok", None)
    if ok is True or passed is True or is_ok is True:
        return tuple()

    # Last resort: treat report as a single “violation-like” object
    return (report,)


def materialize_episode_from_recommendation(
    *,
    store: ArtifactStore,
    orientation: Orientation,
    recommendation: Recommendation,
    raw_inputs: tuple[RawInput, ...] = (),
) -> DecisionEpisode:
    """
    Kernel-owned materialization entrypoint.

    Contract:
    - Providers may propose artifacts, but cannot create canonical binders.
    - The kernel persists Orientation + Recommendation + DecisionEpisode binder and validates.

    Preconditions:
    - Recommendation and referenced artifacts are already in `store` OR persisted by caller.

    Postconditions:
    - Orientation persisted.
    - Recommendation persisted.
    - DecisionEpisode persisted (IDs-only binder).
    - Episode validates; otherwise raises ValueError with violations.
    """
    # Persist anchors (idempotent under sane store semantics)
    _persist_one(store, orientation)
    _persist_one(store, recommendation)
    if raw_inputs:
        _persist_many(store, raw_inputs)

    # Build binder via append-only helpers (matches your dataclass design)
    episode = DecisionEpisode()
    episode = episode.add_orientations(orientation.orientation_id)
    episode = episode.add_recommendations(recommendation.recommendation_id)

    # Thread through provenance IDs if we have them (good for indexing/audit)
    if getattr(recommendation, "evidence_ids", None):
        episode = episode.add_evidence(*tuple(recommendation.evidence_ids))
    if getattr(recommendation, "observation_ids", None):
        episode = episode.add_observations(*tuple(recommendation.observation_ids))
    if getattr(recommendation, "interpretation_ids", None):
        episode = episode.add_interpretations(*tuple(recommendation.interpretation_ids))

    # Options referenced by ranked_options
    ranked = getattr(recommendation, "ranked_options", None)
    if ranked:
        option_ids = tuple(ro.option_id for ro in ranked if getattr(ro, "option_id", None))
        if option_ids:
            episode = episode.add_options(*option_ids)

    # Raw inputs (if supplied)
    if raw_inputs:
        episode = episode.add_raw_inputs(*tuple(ri.raw_input_id for ri in raw_inputs))

    # Persist binder
    _persist_one(store, episode)

    # Validate immediately (normalize ValidationReport -> tuple[violations])
    report = validate_episode(store=store, episode_id=episode.episode_id)
    violations = _normalize_violations(report)

    if violations:
        msgs = []
        for vv in violations:
            rule = getattr(vv, "rule", None) or getattr(vv, "code", None) or "violation"
            msg = getattr(vv, "message", None) or str(vv)
            msgs.append(f"{rule}: {msg}")
        raise ValueError("Episode validation failed:\n" + "\n".join(msgs))

    return episode
