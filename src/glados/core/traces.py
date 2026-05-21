"""JSONL trace per session. Each turn appends to the session's file
(`"a"` mode) so multi-turn sessions accumulate one record per file
rather than truncating on every re-open. Flushed eagerly so a crash
mid-turn still leaves a readable record."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def event(self, kind: str, /, **fields: Any) -> None:
        line = json.dumps({"ts": time.time(), "event": kind, **fields}, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def open(self, session_id: str) -> TraceWriter:
        safe = session_id.replace(":", "_")
        return TraceWriter(self.root / f"{safe}.jsonl")
