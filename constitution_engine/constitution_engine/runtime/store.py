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
class ArtifactStore(Protocol):
    """
    Minimal store interface for kernel artifacts.

    - Objects are stored by (type_name, id)
    - This avoids requiring a shared base class across dataclasses
    """

    def put(self, obj: Any) -> str:
        """Persist an artifact; returns its primary ID."""
        ...

    def get(self, cls: Type[T], obj_id: str) -> Optional[T]:
        """Fetch by class and ID, or None if missing."""
        ...

    def must_get(self, cls: Type[T], obj_id: str) -> T:
        """Fetch by class and ID; raises KeyError if missing."""
        ...

    def has(self, cls: Type[Any], obj_id: str) -> bool:
        """True if present."""
        ...

    def list_ids(self, cls: Type[Any]) -> Sequence[str]:
        """All IDs stored for a class."""
        ...

    def resolve_many(self, cls: Type[T], ids: Iterable[str]) -> Tuple[Sequence[T], Sequence[ResolveError]]:
        """Resolve many IDs; returns (found, errors)."""
        ...
