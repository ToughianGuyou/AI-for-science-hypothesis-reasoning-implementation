"""Append-only, secret-free JSONL trace writing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from hypothesis_reasoning.llm.types import TraceEvent


class TraceWriter:
    """Flush each complete trace event as one standards-compliant JSON line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def append(self, event: TraceEvent) -> None:
        serialized = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
