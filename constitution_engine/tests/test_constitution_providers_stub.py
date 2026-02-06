# constitution_engine/tests/test_constitution_providers_stub.py
from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from importlib import import_module
from typing import Any, get_args, get_origin, get_type_hints


def _import_first(paths: list[str], symbol: str) -> Any:
    """
    Import `symbol` from the first module path that exists.
    Raises a helpful error if none match.
    """
    errors: list[str] = []
    for mod_path in paths:
        try:
            mod = import_module(mod_path)
            if hasattr(mod, symbol):
                return getattr(mod, symbol)
            errors.append(f"{mod_path}: no symbol {symbol}")
        except Exception as e:
            errors.append(f"{mod_path}: {type(e).__name__}: {e}")

    raise ImportError(
        f"Could not import {symbol}. Tried:\n"
        + "\n".join(f"  - {line}" for line in errors)
        + "\n\nFix by updating the candidate module paths in this test to match your repo."
    )


# ---- Import provider-layer symbols (robust to package location) ----
EpisodeContext = _import_first(
    [
        "constitution_providers.context",
        "constitution_engine.constitution_providers.context",
    ],
    "EpisodeContext",
)

ProposalProvider = _import_first(
    [
        "constitution_providers.protocol",
        "constitution_engine.constitution_providers.protocol",
    ],
    "ProposalProvider",
)

StubProvider = _import_first(
    [
        "constitution_providers.stub_provider",
        "constitution_engine.constitution_providers.stub_provider",
    ],
    "StubProvider",
)

# ---- Import kernel types ----
Orientation = _import_first(
    [
        "constitution_engine.models.orientation",
        "constitution_engine.models",  # exported from models/__init__.py in your repo
    ],
    "Orientation",
)

RawInput = _import_first(
    [
        "constitution_engine.models.raw_input",
        "constitution_engine.models",  # exported from models/__init__.py if present
    ],
    "RawInput",
)


def _default_for_type(tp: Any) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[T] / Union[..., None]
    if origin is None and args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != len(args):
            return None

    if origin in (list, tuple, set, frozenset):
        return origin()
    if origin in (dict,):
        return {}

    if tp is str:
        return "x"
    if tp is int:
        return 1
    if tp is float:
        return 0.1
    if tp is bool:
        return False

    if isinstance(tp, type) and is_dataclass(tp):
        return _make_minimal(tp)

    return None


def _make_minimal(cls: type[Any], **overrides: Any) -> Any:
    if not is_dataclass(cls):
        raise TypeError(
            f"{cls.__module__}.{cls.__name__} is not a dataclass. "
            f"Update this test to construct it using your actual constructor."
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for f in fields(cls):
        if f.name in overrides:
            kwargs[f.name] = overrides[f.name]
            continue

        if f.default is not MISSING:
            kwargs[f.name] = f.default
            continue
        if f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
            continue

        name = f.name.lower()
        if name.endswith("_id") or name in {"id"}:
            kwargs[f.name] = "id_x"
            continue
        if name in {"title", "name"}:
            kwargs[f.name] = "title_x"
            continue
        if name in {"description", "text", "prompt"}:
            kwargs[f.name] = "desc_x"
            continue

        tp = hints.get(f.name, Any)
        kwargs[f.name] = _default_for_type(tp)

    return cls(**kwargs)


def test_stub_provider_is_protocol() -> None:
    p = StubProvider()
    assert isinstance(p, ProposalProvider)


def test_stub_provider_proposes_probe_option_and_ranking() -> None:
    orientation = _make_minimal(Orientation)
    raw = _make_minimal(RawInput)

    # IMPORTANT: this expects EpisodeContext(orientation=..., raw_inputs=...)
    ctx = EpisodeContext(orientation=orientation, raw_inputs=(raw,))
    out = StubProvider().propose(ctx)

    assert out.provider_id == "stub_provider"

    assert len(out.evidence) == 0
    assert len(out.observations) == 0
    assert len(out.interpretations) == 0

    assert len(out.options) == 1
    opt = out.options[0]

    # NEW CONTRACT: provider emits ranking inputs only (no Recommendation object)
    assert hasattr(out, "proposed_ranked_options")
    ranked = getattr(out, "proposed_ranked_options", ()) or ()
    assert len(ranked) == 1

    ro = ranked[0]
    assert ro.option_id == opt.option_id
    assert ro.rank == 1

    # Provider should include a rationale for its proposed ranking
    assert getattr(out, "proposed_rationale", None) is not None
    assert isinstance(out.proposed_rationale, str)
    assert out.proposed_rationale.strip() != ""

    # ActionClass should still be probe for the stub option
    assert getattr(opt, "action_class", None) == "probe"
