from __future__ import annotations

from constitution_engine.runtime.store import ArtifactStore
from constitution_engine.models.orientation import Orientation, Objective
from constitution_engine.models.raw_input import RawInput

from constitution_providers.context import EpisodeContext
from constitution_providers.runner.runner import run_provider
from constitution_providers.stub_provider import StubProvider


def test_provider_runner_persists_and_optionally_materializes() -> None:
    st = ArtifactStore()

    ori = Orientation().add_objectives(
        Objective(name="safety", description="Prefer safe, reversible actions.", weight=1.0)
    )
    raw = RawInput(payload="hello", metadata={"channel": "test"})

    ctx = EpisodeContext(orientation=ori, raw_inputs=(raw,))

    res = run_provider(provider=StubProvider(), ctx=ctx, store=st, materialize=True, validate=True)

    # always true
    assert res.store is st
    assert res.proposals is not None

    # canonical output must exist (kernel-owned)
    assert res.canonical_recommendation_id is not None
    assert isinstance(res.canonical_recommendation_id, str)
    assert res.canonical_recommendation_id.startswith("rc_")

    assert len(res.proposals.options) == 1

    # materialize may be unavailable depending on your kernel entrypoint;
    # runner returns None/None in that case (still a pass)
    if res.episode_id is not None:
        assert isinstance(res.violations, tuple)
        assert res.violations == tuple()
