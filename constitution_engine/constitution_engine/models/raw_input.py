#constitution_engine/constitution_engine/models/raw_input.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

from .types import new_id, now_utc


@dataclass(frozen=True)
class RawInput:
    """
    Immutable record of what entered the system.

    - payload: unmodified user/system input (text, json, etc.)
    - metadata: channel, actor, request_id, etc.
    """
    raw_input_id: str = field(default_factory=lambda: new_id("raw"))
    created_at: datetime = field(default_factory=now_utc)

    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # Optional: reference to a parent raw input if this is a derived ingestion step
    parent_raw_input_id: Optional[str] = None
