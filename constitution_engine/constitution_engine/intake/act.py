from __future__ import annotations

from dataclasses import replace

from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.option import Option
from constitution_engine.models.choice import ChoiceRecord
from constitution_engine.models.types import now_utc, new_id
from constitution_engine.runtime.store import ArtifactStore


def _append_unique(seq: tuple[str, ...], *items: str) -> tuple[str, ...]:
    out = list(seq)
    seen = set(out)
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return tuple(out)


def choose(
    *,
    store: ArtifactStore,
    episode_id: str,
    recommendation_id: str | None = None,
    chosen_option_id: str,
    used_override: bool = False,
    rationale: str | None = None,
) -> str:
    """
    Kernel entrypoint: Recommend → Choose → Act marker.

    What it does (thin-slice, auditable, deterministic):
      1) Ensures the episode and option exist, and option is part of the episode.
      2) Resolves/infer recommendation_id (strict: must exist and be part of episode).
      3) Creates a ChoiceRecord linking (episode, recommendation, option).
      4) Appends choice_id to episode.choice_ids (ids-only binder) if supported.
      5) Marks episode as acted (episode.mark_acted), setting chosen_option_id and acted_at.

    Returns: choice_id (the new canonical choice artifact id).
    """
    ep = store.must_get(DecisionEpisode, episode_id)

    # Ensure option exists (and is part of the episode)
    _ = store.must_get(Option, chosen_option_id)
    if chosen_option_id not in ep.option_ids:
        raise ValueError(f"Option {chosen_option_id} is not part of episode {episode_id}")

    # Ensure recommendation exists (strict for auditability), but allow inference for back-compat
    from constitution_engine.models.recommendation import Recommendation  # local import

    rid = recommendation_id or ep.latest_recommendation_id()
    if not rid:
        raise ValueError(
            f"Episode {episode_id} has no recommendation_ids; cannot choose without recommendation context."
        )

    _ = store.must_get(Recommendation, rid)
    if rid not in ep.recommendation_ids:
        raise ValueError(f"Recommendation {rid} is not part of episode {episode_id}")

    ts = now_utc()

    choice = ChoiceRecord(
        choice_id=new_id("ch"),
        episode_id=ep.episode_id,
        recommendation_id=rid,
        option_id=chosen_option_id,
        used_override=bool(used_override),
        rationale=(rationale or "").strip(),
        created_at=ts,
    )
    store.put(choice)

    # Update episode binder + act marker
    ep2 = ep

    # Append choice id if this DecisionEpisode version supports choice_ids
    if hasattr(ep2, "choice_ids"):
        current = tuple(getattr(ep2, "choice_ids", tuple()) or tuple())
        ep2 = replace(ep2, choice_ids=_append_unique(current, choice.choice_id))

    # Mark acted (canonical helper if present; else set fields directly)
    if hasattr(ep2, "mark_acted"):
        ep2 = ep2.mark_acted(chosen_option_id=chosen_option_id, acted_at=ts)
    else:
        ep2 = replace(ep2, acted=True, chosen_option_id=chosen_option_id, acted_at=ts)

    store.put(ep2)
    return choice.choice_id


# Backwards-compatible alias (if older callers used act_on_option).
def act_on_option(
    *,
    store: ArtifactStore,
    episode_id: str,
    chosen_option_id: str,
    recommendation_id: str | None = None,
    used_override: bool = False,
    rationale: str | None = None,
) -> str:
    """
    Legacy-friendly wrapper.
    Prefer `choose(...)` going forward.

    Returns: choice_id
    """
    return choose(
        store=store,
        episode_id=episode_id,
        recommendation_id=recommendation_id,
        chosen_option_id=chosen_option_id,
        used_override=used_override,
        rationale=rationale,
    )
