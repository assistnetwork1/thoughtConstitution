import sys
from pathlib import Path

# -------------------------------------------------------------------
# Make repo root importable BEFORE importing any project packages.
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from constitution_engine.models.types import (  # noqa: E402
    Confidence,
    InfoType,
    new_id,
    now_utc,
)
from constitution_engine.models.observation import Observation  # noqa: E402
from constitution_engine.models.evidence import Evidence, SourceRef  # noqa: E402
from constitution_engine.models.option import Option, OptionKind  # noqa: E402
from constitution_engine.models.recommendation import Recommendation, RankedOption  # noqa: E402
from constitution_engine.models.orientation import Orientation  # noqa: E402

from constitution_providers.context import EpisodeContext  # noqa: E402


@pytest.fixture
def ctx() -> EpisodeContext:
    """
    Minimal EpisodeContext fixture for provider/LLM smoke tests.

    - Contains Orientation
    - Contains at least 1 Evidence item so ProposalSet.evidence_refs can resolve
    """
    ori = Orientation(meta={"mode": "test"})

    ev = Evidence(
        evidence_id=new_id("ev"),
        created_at=now_utc(),
        sources=(SourceRef(uri="stub://source", extra={"mode": "test"}),),
        spans=tuple(),
        summary="Stub evidence summary.",
        notes={"mode": "test"},
        integrity=Confidence(1.0),
    )

    return EpisodeContext(
        orientation=ori,
        raw_inputs=tuple(),
        evidence=(ev,),
        meta={"mode": "test"},
    )


@pytest.fixture
def make_minimal_bundle():
    """
    Returns a factory function that builds a *valid baseline* bundle,
    then lets each test override one field (e.g., ranked_options=[]).
    """

    def _make(
        *,
        ranked_options=None,
        recommendation_provenance: bool = True,
        evidence_id: str = "ev1",
        observation_id: str = "obs1",
        option_id: str = "opt1",
        recommendation_id: str = "rec1",
        orientation_id: str = "ori1",
        raw_input_id: str = "ri1",
    ):
        # ---------- Evidence ----------
        ev = Evidence(
            evidence_id=evidence_id,
            sources=(SourceRef(uri="https://example.com"),),
        )
        evidence_items = [ev]

        # ---------- Observations ----------
        obs = Observation(
            observation_id=observation_id,
            info_type=InfoType.FACT,
            raw_input_ids=(raw_input_id,),
            evidence_ids=(evidence_id,),
        )
        observations = [obs]

        # ---------- Options ----------
        opt = Option(
            option_id=option_id,
            kind=OptionKind.INFO_GATHERING,
            observation_ids=(observation_id,),
            evidence_ids=(evidence_id,),
            interpretation_ids=tuple(),
            orientation_id=orientation_id,
            uncertainties=tuple(),
        )
        options = [opt]

        # ---------- Recommendation ----------
        if ranked_options is None:
            ranked_options = [
                RankedOption(
                    option_id=option_id,
                    rank=1,
                    score=0.0,
                    rationale="fixture",
                )
            ]

        rec = Recommendation(
            recommendation_id=recommendation_id,
            orientation_id=orientation_id,
            ranked_options=tuple(ranked_options),
            evidence_ids=(evidence_id,) if recommendation_provenance else tuple(),
            observation_ids=(observation_id,) if recommendation_provenance else tuple(),
            interpretation_ids=tuple(),
            model_state_ids=tuple(),
        )

        return observations, evidence_items, options, rec

    return _make
