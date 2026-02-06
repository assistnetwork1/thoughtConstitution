# tests/test_runner_multi_thread_evidence.py
from __future__ import annotations

from constitution_engine.models.orientation import Orientation
from constitution_engine.models.raw_input import RawInput

from constitution_providers.context import EpisodeContext
from constitution_providers.runner.runner_multi import run_providers
from constitution_providers.retriever_stub import StubRetrieverProvider
from constitution_providers.stub_provider import StubProvider


def test_runner_multi_threads_evidence_into_reasoner_and_persists_provenance_ids() -> None:
    """
    Contract test for the multi-provider "evidence bus":

    Expectation:
      - Retriever emits Evidence artifacts.
      - runner_multi threads Evidence into downstream providers via ctx.evidence.
      - Reasoner (StubProvider) binds emitted evidence_ids onto its Option.
      - Canonical Recommendation includes evidence_ids (derived from merged evidence).

    This locks the key constitutional behavior:
      retrieve -> reason -> link provenance -> canonicalize
    """
    orientation = Orientation(
        objectives=("Decide next step for a new service idea",),
        constraints=(),
        values=(),
        meta={"mode": "test"},
    )

    raw_inputs = (
        RawInput(
            raw_input_id="ri_test_1",
            payload="We can bring dogs to offices to reduce stress.",
            metadata={"source": "user"},
        ),
    )

    # IMPORTANT: ctx.evidence is intentionally omitted here (defaults to empty in your EpisodeContext)
    ctx = EpisodeContext(
        orientation=orientation,
        raw_inputs=raw_inputs,
        meta={"mode": "test"},
    )

    result = run_providers(
        providers=[StubRetrieverProvider(), StubProvider()],
        ctx=ctx,
        thread_evidence=True,
        materialize=True,
        validate=True,
    )

    # Sanity: run produced no violations
    assert result.violations == tuple()

    # We should have two ProposalSets: retriever then reasoner
    assert len(result.proposals) == 2
    retr_ps, reason_ps = result.proposals

    # Retriever produced exactly one Evidence
    assert len(retr_ps.evidence) == 1
    evidence = retr_ps.evidence[0]
    assert hasattr(evidence, "evidence_id")
    evidence_id = getattr(evidence, "evidence_id")
    assert isinstance(evidence_id, str) and evidence_id.startswith("ev_")

    # Reasoner produced exactly one Option, and it should reference the evidence_id
    assert len(reason_ps.options) == 1
    opt = reason_ps.options[0]
    assert hasattr(opt, "evidence_ids")
    assert evidence_id in getattr(opt, "evidence_ids")

    # Canonical recommendation should have been created and materialized
    assert result.canonical_recommendation_id is not None
    assert isinstance(result.canonical_recommendation_id, str)
    assert result.canonical_recommendation_id.startswith("rc_")
    assert result.episode_id is not None
    assert result.orientation_id == orientation.orientation_id

    # Recommendation must include evidence_id (derived from merged evidence)
    rec = result.store.get(Recommendation, result.canonical_recommendation_id)  # type: ignore[name-defined]
    assert evidence_id in rec.evidence_ids


# Local import to avoid hard dependency at module import time
from constitution_engine.models.recommendation import Recommendation  # noqa: E402
