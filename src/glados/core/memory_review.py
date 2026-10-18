"""Shadow-mode review file for per-speaker memory (DESIGN-speaker-memory.md).

While `[memory] inject = false`, nothing the batch writes reaches a turn.
Instead each change it makes is appended here as one JSON line with an empty
`score`, so a human can judge whether the facts would have helped before any
of them is fed back. The file holds fact text, so it lives outside the repo,
created mode 0600. On Windows that mode only toggles read-only: there the
profile directory's ACL is what keeps it private, as for the database.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class ReviewLog:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def append(self, entries: list[dict]) -> None:
        """Blocking: call it off the loop."""
        if not entries:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps({"at": stamp, **entry, "score": None}) + "\n")
        _restrict_to_owner(self._path)


def _restrict_to_owner(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        log.warning("memory review: could not restrict file mode on %s", path.name)
