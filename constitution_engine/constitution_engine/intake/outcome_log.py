from __future__ import annotations

from dataclasses import replace

from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.outcome import Outcome
from constitution_engine.models.types import Confidence, new_id, now_utc
from constitution_engine.runtime.store import ArtifactStore


def log_outcome(
    *,
    store: ArtifactStore,
    episode_id: str,
    recommendation_id: str | None,
    chosen_option_id: str | None,
    description: str,
) -> str:
    """
    Create and store an Outcome, attach it to the episode (episode.outcome_ids).
    Returns outcome_id.
    """
    ep = store.must_get(DecisionEpisode, episode_id)

    out = Outcome(
        outcome_id=new_id("out"),
        created_at=now_utc(),
        recommendation_id=recommendation_id,
        chosen_option_id=chosen_option_id,
        description=description.strip(),
        confidence=Confidence(0.6),
        uncertainties=(),
        evidence_ids=(),
        meta={},
    )
    store.put(out)

    ep2 = ep.add_outcomes(out.outcome_id)
    store.put(ep2)
    return out.outcome_id
