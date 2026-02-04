from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from constitution_engine.models.raw_input import RawInput
from constitution_engine.models.evidence import Evidence
from constitution_engine.models.observation import Observation
from constitution_engine.models.interpretation import Interpretation
from constitution_engine.models.orientation import Orientation
from constitution_engine.models.option import Option
from constitution_engine.models.recommendation import Recommendation
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.runtime.store import ArtifactStore


@dataclass(frozen=True)
class EngineConfig:
    """
    Kernel runtime configuration.
    Keep small; domain policies live outside.
    """
    auto_create_episode: bool = True


class Engine:
    """
    Orchestrates Observe → Model → Orient → Act → Review.

    Domain logic is injected as callables that return kernel artifacts.
    The engine stores them and maintains a DecisionEpisode index.
    """

    def __init__(self, store: ArtifactStore, config: Optional[EngineConfig] = None) -> None:
        self.store = store
        self.config = config or EngineConfig()

    def run(
        self,
        *,
        raw_input: RawInput,
        evidence: Optional[Evidence] = None,
        observe: Optional[Callable[[RawInput, Optional[Evidence]], Sequence[Observation]]] = None,
        interpret: Optional[Callable[[Sequence[Observation], Optional[Evidence]], Sequence[Interpretation]]] = None,
        orient: Optional[Callable[[RawInput, Sequence[Observation], Sequence[Interpretation]], Orientation]] = None,
        propose_options: Optional[
            Callable[[Orientation, Sequence[Observation], Sequence[Interpretation]], Sequence[Option]]
        ] = None,
        recommend: Optional[
            Callable[[Orientation, Sequence[Option], Sequence[Observation], Sequence[Interpretation]], Recommendation]
        ] = None,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
    ) -> DecisionEpisode:
        # Store ingress
        raw_id = self.store.put(raw_input)

        ev_id = None
        if evidence is not None:
            ev_id = self.store.put(evidence)

        # Observe
        observations: Sequence[Observation] = ()
        if observe is not None:
            observations = observe(raw_input, evidence)
        obs_ids = tuple(self.store.put(o) for o in observations)

        # Interpret
        interpretations: Sequence[Interpretation] = ()
        if interpret is not None:
            interpretations = interpret(observations, evidence)
        int_ids = tuple(self.store.put(i) for i in interpretations)

        # Orient (required for action/recommendation)
        orientation: Optional[Orientation] = None
        ori_id: Optional[str] = None
        if orient is not None:
            orientation = orient(raw_input, observations, interpretations)
            ori_id = self.store.put(orientation)

        # Options
        options: Sequence[Option] = ()
        if (propose_options is not None) and (orientation is not None):
            options = propose_options(orientation, observations, interpretations)
        opt_ids = tuple(self.store.put(o) for o in options)

        # Recommendation
        rec_id: Optional[str] = None
        if (recommend is not None) and (orientation is not None):
            rec = recommend(orientation, options, observations, interpretations)
            rec_id = self.store.put(rec)

        # Episode index
        episode = DecisionEpisode(
            title=episode_title,
            description=episode_description,
            raw_input_ids=(raw_id,),
            evidence_ids=((ev_id,) if ev_id else ()),
            observation_ids=obs_ids,
            interpretation_ids=int_ids,
            orientation_ids=((ori_id,) if ori_id else ()),
            option_ids=opt_ids,
            recommendation_ids=((rec_id,) if rec_id else ()),
        )
        self.store.put(episode)
        return episode
