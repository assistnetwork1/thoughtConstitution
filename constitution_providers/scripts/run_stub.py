# constitution_providers/scripts/run_stub.py
from __future__ import annotations

from constitution_engine.models.orientation import Orientation
from constitution_engine.models.raw_input import RawInput
from constitution_providers.context import EpisodeContext
from constitution_providers.runner import run_provider
from constitution_providers.stub_provider import StubProvider


def main() -> None:
    orientation = Orientation(
        objectives=("Decide next step for a new service idea",),
        constraints=(),
        values=(),
        meta={"mode": "demo"},
    )

    raw_inputs = (
        RawInput(
            raw_input_id="ri_demo_1",
            payload="We can bring dogs to offices to reduce stress.",
            metadata={"source": "user"},
        ),
    )

    ctx = EpisodeContext(
        orientation=orientation,
        raw_inputs=raw_inputs,
        meta={"mode": "demo"},
    )

    result = run_provider(provider=StubProvider(), ctx=ctx)
    print(result)


if __name__ == "__main__":
    main()
