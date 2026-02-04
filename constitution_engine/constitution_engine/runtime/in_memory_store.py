from __future__ import annotations

from dataclasses import is_dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type, TypeVar

from .store import ArtifactStore, ResolveError

T = TypeVar("T")


def _type_name(cls: Type[Any]) -> str:
    return cls.__name__


def _infer_primary_id(obj: Any) -> str:
    """
    Finds the primary ID field by convention:
      - first dataclass field ending with "_id"
      - or KeyError if none found
    """
    if not is_dataclass(obj):
        raise TypeError("Only dataclass artifacts are supported by InMemoryArtifactStore")

    for f in obj.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.name.endswith("_id"):
            value = getattr(obj, f.name, None)
            if isinstance(value, str) and value:
                return value
    raise KeyError(f"Could not infer primary id field for {type(obj).__name__}")


class InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self._lock = RLock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def put(self, obj: Any) -> str:
        cls = type(obj)
        type_name = _type_name(cls)
        obj_id = _infer_primary_id(obj)

        with self._lock:
            self._data.setdefault(type_name, {})
            self._data[type_name][obj_id] = obj
        return obj_id

    def get(self, cls: Type[T], obj_id: str) -> Optional[T]:
        type_name = _type_name(cls)
        with self._lock:
            obj = self._data.get(type_name, {}).get(obj_id)
        return obj  # type: ignore[return-value]

    def must_get(self, cls: Type[T], obj_id: str) -> T:
        obj = self.get(cls, obj_id)
        if obj is None:
            raise KeyError(f"{_type_name(cls)} not found: {obj_id}")
        return obj

    def has(self, cls: Type[Any], obj_id: str) -> bool:
        return self.get(cls, obj_id) is not None

    def list_ids(self, cls: Type[Any]) -> Sequence[str]:
        type_name = _type_name(cls)
        with self._lock:
            return tuple(self._data.get(type_name, {}).keys())

    def resolve_many(self, cls: Type[T], ids: Iterable[str]) -> Tuple[Sequence[T], Sequence[ResolveError]]:
        found = []
        errors = []
        for obj_id in ids:
            obj = self.get(cls, obj_id)
            if obj is None:
                errors.append(ResolveError(artifact_type=_type_name(cls), artifact_id=obj_id))
            else:
                found.append(obj)
        return tuple(found), tuple(errors)
