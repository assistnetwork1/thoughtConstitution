# constitution_engine/scripts/quick_sim.py
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

from constitution_engine.invariants.validate import validate_episode
from constitution_engine.models.episode import DecisionEpisode
from constitution_engine.models.evidence import Evidence, SourceRef
from constitution_engine.models.observation import Observation
from constitution_engine.models.option import Option, OptionKind
from constitution_engine.models.recommendation import Recommendation, RankedOption
from constitution_engine.models.review import ReviewRecord
from constitution_engine.models.types import InfoType
from constitution_engine.runtime.store import ArtifactStore


def _make(cls: type[Any], **kwargs: Any) -> Any:
    """
    Create dataclass instances defensively by filtering unknown kwargs.
    This makes the script resilient while models are still evolving.
    """
    if is_dataclass(cls):
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)
    return cls(**kwargs)


def _set_if_supported(obj: Any, *, method: str, field_name: str, value: Any) -> Any:
    """
    Prefer immutable helper methods (with_kind / with_action_class / etc).
    Fall back to dataclasses.replace if the field exists.
    """
    if hasattr(obj, method):
        return getattr(obj, method)(value)

    if is_dataclass(obj):
        names = {f.name for f in fields(obj)}
        if field_name in names:
            return replace(obj, **{field_name: value})

    return obj


def main() -> None:
    store = ArtifactStore()

    # --- Evidence ---
    ev = _make(
        Evidence,
        evidence_id="ev1",
        sources=(
            _make(
                SourceRef,
                uri="file://local/demo.txt",
                title="Demo Source",
                author="quick_sim",
            ),
        ),
        summary="Minimal evidence bundle for quick simulation.",
    )

    # --- Observation ---
    # We avoid assuming too many fields; set only what is common/likely.
    obs = _make(
        Observation,
        observation_id="obs1",
        info_type=InfoType.FACT,
        evidence_ids=("ev1",),
    )

    # --- Option ---
    # Use PROBE action_class to avoid gate failures while you’re iterating.
    opt = _make(Option, option_id="opt1")
    opt = _set_if_supported(opt, method="with_kind", field_name="kind", value=OptionKind.EXECUTE)
    opt = _set_if_supported(opt, method="with_action_class", field_name="action_class", value="probe")
    opt = _set_if_supported(opt, method="with_orientation_id", field_name="orientation_id", value="ori1")

    # --- Recommendation ---
    rec = _make(
        Recommendation,
        recommendation_id="rec1",
        orientation_id="ori1",
        ranked_options=(RankedOption(option_id="opt1", rank=1, score=0.5, rationale="Start with a probe."),),
        evidence_ids=("ev1",),
        observation_ids=("obs1",),
    )

    # --- Review (optional here; included to exercise wiring) ---
    rev = _make(
        ReviewRecord,
        review_id="rev1",
        episode_id="ep1",
        override_audit={"overrides": []},
    )

    # --- Episode ---
    ep = _make(
        DecisionEpisode,
        episode_id="ep1",
        title="Quick Sim Episode",
        description="Minimal Observe→Recommend→(optional)Review loop",
        evidence_ids=("ev1",),
        observation_ids=("obs1",),
        recommendation_ids=("rec1",),
        review_ids=("rev1",),
    )

    # Store everything
    store.put(ev)
    store.put(obs)
    store.put(opt)
    store.put(rec)
    store.put(rev)
    store.put(ep)

    # Validate episode
    report = validate_episode(store, "ep1")

    print(f"\n== {report.subject} ==")
    print(f"OK: {report.ok}")
    if report.resolve_errors:
        print("\nResolve errors:")
        for e in report.resolve_errors:
            print(f"  - {e.artifact_type}:{e.artifact_id} ({e.message})")

    if report.violations:
        print("\nViolations:")
        for v in report.violations:
            print(f"  - {v.rule}: {v.message}")
    else:
        print("\nViolations: (none)")


if __name__ == "__main__":
    main()
