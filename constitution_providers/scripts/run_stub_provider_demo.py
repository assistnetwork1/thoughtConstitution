from __future__ import annotations

from constitution_engine.models.orientation import Orientation, Objective
from constitution_engine.models.raw_input import RawInput
from constitution_engine.runtime.store import ArtifactStore

from constitution_providers.context import EpisodeContext
from constitution_providers.runner import run_provider
from constitution_providers.stub_provider import StubProvider


def main() -> None:
    st = ArtifactStore()

    ori = Orientation().add_objectives(
        Objective(name="safety", description="Prefer safe, reversible actions.", weight=1.0)
    )
    raw = RawInput(payload="Need to decide but inputs are incomplete.", metadata={"channel": "demo"})

    ctx = EpisodeContext(orientation=ori, raw_inputs=(raw,))
    res = run_provider(provider=StubProvider(), ctx=ctx, store=st, materialize=True, validate=True)

    print("provider:", "stub_provider")
    print("recommendation_id:", res.recommendation_id)
    print("episode_id:", res.episode_id)
    print("orientation_id:", res.orientation_id)
    print("violations:", res.violations)


if __name__ == "__main__":
    main()
