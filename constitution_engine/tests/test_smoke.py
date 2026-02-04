# tests/test_smoke.py

from constitution_engine.models import RawInput
from constitution_engine.runtime import InMemoryArtifactStore, Engine
from constitution_engine.invariants.validate import validate_episode


def test_minimal_episode_smoke():
    # 1. Create a store (acts like your database)
    store = InMemoryArtifactStore()

    # 2. Create the engine with that store
    engine = Engine(store)

    # 3. Run the engine with ONLY raw input (no domain logic)
    episode = engine.run(
        raw_input=RawInput(payload="hello world"),
        episode_title="smoke test",
    )

    # 4. Validate the resulting episode
    report = validate_episode(store, episode.episode_id)

    # 5. Assert the constitution is satisfied
    assert report.ok, f"Violations: {report.violations}, Resolve errors: {report.resolve_errors}"
