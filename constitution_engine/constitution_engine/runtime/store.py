from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple, Type, TypeVar, runtime_checkable

T = TypeVar("T")


@dataclass(frozen=True)
class ResolveError:
    artifact_type: str
    artifact_id: str
    message: str = "not found"


@runtime_checkable
class ArtifactStoreProtocol(Protocol):
    """
    Minimal store interface for kernel artifacts.

    - Objects are stored by (type_key, id)
    - type_key is module+qualname so it is stable and avoids class-identity pitfalls.
    """

    def put(self, obj: Any) -> str:
        ...

    def get(self, cls: Type[T], obj_id: str) -> Optional[T]:
        ...

    def must_get(self, cls: Type[T], obj_id: str) -> T:
        ...

    def has(self, cls: Type[Any], obj_id: str) -> bool:
        ...

    def list_ids(self, cls: Type[Any]) -> Sequence[str]:
        ...

    def resolve_many(self, cls: Type[T], ids: Iterable[str]) -> Tuple[Sequence[T], Sequence[ResolveError]]:
        ...


def _type_key(cls: Type[Any]) -> str:
    # Stable key across imports: module + qualname
    return f"{cls.__module__}.{cls.__qualname__}"


def _infer_primary_id(obj: Any) -> str:
    """
    Heuristic ID inference (deterministic):
    1) Prefer "<classname_lower>_id" when present (e.g. Recommendation -> recommendation_id)
    2) Prefer explicit common primaries (review_id, recommendation_id, option_id, orientation_id)
       before episode_id to avoid mis-keying ReviewRecord by episode_id.
    3) If exactly one "*_id" attribute exists and is a str, use it.
    4) If multiple candidates exist, prefer shortest attr name (then lexicographic) deterministically.
    """
    cls_name = type(obj).__name__.lower()
    preferred = f"{cls_name}_id"

    # 1) canonical "<classname>_id"
    if hasattr(obj, preferred):
        val = getattr(obj, preferred, None)
        if isinstance(val, str) and val:
            return val

    # 2) common primaries (avoid episode_id stealing ReviewRecord)
    for attr in ("review_id", "recommendation_id", "option_id", "orientation_id"):
        if hasattr(obj, attr):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val:
                return val

    # DecisionEpisode primary
    if hasattr(obj, "episode_id"):
        val = getattr(obj, "episode_id", None)
        if isinstance(val, str) and val:
            return val

    # 3) fallback: find any *_id string fields
    candidates: list[Tuple[str, str]] = []
    for attr in dir(obj):
        if not attr.endswith("_id"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(val, str) and val:
            candidates.append((attr, val))

    if len(candidates) == 1:
        return candidates[0][1]

    # 4) deterministic pick if multiple
    if candidates:
        candidates.sort(key=lambda t: (len(t[0]), t[0]))
        return candidates[0][1]

    raise ValueError(f"Cannot infer primary id for object of type {type(obj)!r}")


class InMemoryArtifactStore:
    """
    Concrete in-memory ArtifactStore used by tests and the thin-slice runner.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Any] = {}
        self._ids_by_type: Dict[str, list[str]] = {}

    def put(self, obj: Any) -> str:
        obj_id = _infer_primary_id(obj)
        tk = _type_key(type(obj))
        key = (tk, obj_id)

        self._data[key] = obj

        ids = self._ids_by_type.setdefault(tk, [])
        if obj_id not in ids:
            ids.append(obj_id)

        return obj_id

    def get(self, cls: Type[T], obj_id: str) -> Optional[T]:
        tk = _type_key(cls)
        obj = self._data.get((tk, obj_id))
        return obj  # type: ignore[return-value]

    def must_get(self, cls: Type[T], obj_id: str) -> T:
        obj = self.get(cls, obj_id)
        if obj is None:
            raise KeyError(obj_id)
        return obj

    def has(self, cls: Type[Any], obj_id: str) -> bool:
        return self.get(cls, obj_id) is not None

    def list_ids(self, cls: Type[Any]) -> Sequence[str]:
        tk = _type_key(cls)
        return tuple(self._ids_by_type.get(tk, []))

    def resolve_many(self, cls: Type[T], ids: Iterable[str]) -> Tuple[Sequence[T], Sequence[ResolveError]]:
        found: list[T] = []
        errors: list[ResolveError] = []

        for obj_id in ids:
            obj = self.get(cls, obj_id)
            if obj is None:
                errors.append(ResolveError(artifact_type=cls.__name__, artifact_id=obj_id))
            else:
                found.append(obj)

        return tuple(found), tuple(errors)


# What the rest of the codebase imports in tests:
# It must be instantiable, so we export the concrete store here.
ArtifactStore = InMemoryArtifactStore
