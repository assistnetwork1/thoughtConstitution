#constitution_engine/constitution_engine/models/evidence.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .types import new_id, now_utc, Confidence


@dataclass(frozen=True)
class SpanRef:
    """
    Pointer into a source (text spans, timestamps, page numbers, etc.)
    Use what makes sense per medium.
    """
    start: Optional[int] = None
    end: Optional[int] = None
    page: Optional[int] = None
    timestamp_ms: Optional[int] = None


@dataclass(frozen=True)
class SourceRef:
    """
    A source descriptor.
    uri can be a URL, file path, database key, etc.
    """
    uri: str
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    retrieved_at: datetime = field(default_factory=now_utc)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """
    A typed evidence bundle: sources + spans + notes.

    This is the provenance anchor other artifacts can reference by ID.
    """
    evidence_id: str = field(default_factory=lambda: new_id("ev"))
    created_at: datetime = field(default_factory=now_utc)

    sources: Sequence[SourceRef] = field(default_factory=tuple)
    spans: Sequence[SpanRef] = field(default_factory=tuple)

    summary: Optional[str] = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    # Confidence in the evidence integrity (not the claim itself)
    integrity: Confidence = field(default_factory=lambda: Confidence(1.0))
