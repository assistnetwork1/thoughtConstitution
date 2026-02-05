from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Iterable

from constitution_engine.intake.types import DraftEpisode
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.orientation import Orientation
from constitution_engine.models.types import RiskPosture, new_id, now_utc
from constitution_engine.runtime.store import ArtifactStore


def _make(cls: type[Any], **kwargs: Any) -> Any:
    """
    Defensive constructor: only pass fields that exist on the target dataclass.
    This lets the intake/materialize layer survive model evolution.
    """
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)
    return cls(**kwargs)


def _default_orientation(*, goal_statement: str) -> Orientation:
    """
    Minimal Orientation v1:
    - enough to satisfy Recommendation.orientation_id
    - enough to allow gating + overrides to be interpreted later

    NOTE: We do NOT assume GovernanceMode exists. If your Orientation has
    governance fields, they’ll be set only if the fields exist.
    """
    return _make(
        Orientation,
        orientation_id=new_id("ori"),
        created_at=now_utc(),
        objectives=(goal_statement,),
        constraints=(),
        weights={},  # if supported
        risk_posture=RiskPosture.BALANCED,
        governance_mode="advisory_only",  # if supported (string is fine if your enum accepts it)
        override_scope=(),  # if supported
        override_rationale=None,  # if supported
    )


def materialize_draft_episode(
    *,
    store: ArtifactStore,
    draft: DraftEpisode,
) -> str:
    """
    Persist DraftEpisode into ArtifactStore and return the stored episode_id.
    """
    # 1) Create Orientation (v1 default)
    ori = _default_orientation(goal_statement=draft.goal.statement)
    store.put(ori)

    # 2) Store Evidence / Observations / Interpretations / Options
    for ev in draft.evidence:
        store.put(ev)

    for obs in draft.observations:
        store.put(obs)

    for it in draft.interpretations:
        store.put(it)

    for opt in draft.options:
        # If Option has orientation_id, wire it now (safe)
        try:
            opt2 = replace(opt, orientation_id=ori.orientation_id)
        except TypeError:
            opt2 = opt
        store.put(opt2)

    # 3) Store Recommendation (must have orientation_id)
    rec_id: str | None = None
    if draft.recommendation is not None:
        rec = draft.recommendation

        # Some Recommendation models require orientation_id (yours does).
        # Also wire provenance pointers if those fields exist.
        rec2 = _make(
            type(rec),
            **{
                **rec.__dict__,
                "orientation_id": getattr(ori, "orientation_id"),
                "evidence_ids": tuple(ev.evidence_id for ev in draft.evidence),
                "observation_ids": tuple(o.observation_id for o in draft.observations),
                "interpretation_ids": tuple(i.interpretation_id for i in draft.interpretations),
            },
        )
        store.put(rec2)
        rec_id = getattr(rec2, "recommendation_id", None)

    # 4) Store DecisionEpisode (the canonical binder)
    # DraftEpisode may or may not have raw_inputs; keep it optional.
    raw_ids: tuple[str, ...] = ()
    if hasattr(draft, "raw_inputs") and getattr(draft, "raw_inputs") is not None:
        raw_ids = tuple(ri.raw_id for ri in getattr(draft, "raw_inputs"))

    ep = _make(
        DecisionEpisode,
        episode_id=new_id("ep"),
        created_at=now_utc(),
        goal_id=draft.goal.goal_id,
        raw_input_ids=raw_ids,
        evidence_ids=tuple(ev.evidence_id for ev in draft.evidence),
        observation_ids=tuple(o.observation_id for o in draft.observations),
        interpretation_ids=tuple(i.interpretation_id for i in draft.interpretations),
        orientation_id=getattr(ori, "orientation_id"),
        option_ids=tuple(o.option_id for o in draft.options),
        recommendation_ids=(rec_id,) if rec_id else (),
        review_ids=(),
    )
    store.put(ep)

    return getattr(ep, "episode_id")
