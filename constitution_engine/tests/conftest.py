import pytest

from constitution_engine.models.types import InfoType
from constitution_engine.models.observation import Observation
from constitution_engine.models.evidence import Evidence, SourceRef
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import Recommendation, RankedOption


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
    ):
        # ---------- Evidence ----------
        ev = Evidence(
            evidence_id="ev1",
            sources=[SourceRef(uri="https://example.com")],
        )
        evidence_items = [ev]

        # ---------- Observations ----------
        obs = Observation(
            observation_id="obs1",
            info_type=InfoType.FACT,
            raw_input_ids=("ri1",),
            evidence_ids=("ev1",),
        )
        observations = [obs]

        # ---------- Options ----------
        # Baseline option is non-executing to avoid triggering EXECUTE-only invariants.
        # Do NOT pass reversibility/impact=None; let model defaults apply.
        opt = Option(
            option_id="opt1",
            kind=OptionKind.INFO_GATHERING,
            observation_ids=("obs1",),
            evidence_ids=("ev1",),
            interpretation_ids=tuple(),
            orientation_id="ori1",
            uncertainties=tuple(),
        )
        options = [opt]

        # ---------- Recommendation ----------
        if ranked_options is None:
            ranked_options = [
                RankedOption(
                    option_id="opt1",
                    rank=1,
                    score=0.0,
                    rationale="fixture",
                )
            ]

        rec = Recommendation(
            recommendation_id="rec1",
            orientation_id="ori1",
            ranked_options=tuple(ranked_options),
            # provenance pointers:
            evidence_ids=("ev1",) if recommendation_provenance else tuple(),
            observation_ids=("obs1",) if recommendation_provenance else tuple(),
            interpretation_ids=tuple(),
            model_state_ids=tuple(),
        )

        return observations, evidence_items, options, rec

    return _make
