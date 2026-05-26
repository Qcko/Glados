"""Process-wide logging configuration.

GLaDOS uses `logging.getLogger(__name__)` throughout but never configured
the root logger, so anything below WARNING was silently dropped (uvicorn
only forwards its own access/error log). This module installs a rotating
file handler + a stderr handler the first time `setup_logging()` is
called, idempotently.

Log directory comes from `$GLADOS_LOG_DIR` (set this per machine to keep
absolute paths out of the repo). Falls back to `~/.glados/logs/` so the
process still has somewhere to write if the env var is unset.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUPS = 5

_configured = False


def setup_logging() -> Path:
    """Install handlers on the root logger. Returns the log file path.

    Safe to call more than once — only the first call wires handlers.
    Skipped under pytest so test runs don't install a RotatingFileHandler
    on the shared root logger (which would survive across test cases and
    spam glados.log with test noise).
    """
    global _configured
    log_dir = Path(os.environ.get("GLADOS_LOG_DIR") or (Path.home() / ".glados" / "logs"))
    log_path = log_dir / "glados.log"
    if _configured or "pytest" in sys.modules:
        return log_path

    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Drop pre-existing handlers (uvicorn installs its own that double-print
    # every line we then add). The uvicorn access logger keeps its own
    # handlers because we only touch the root logger.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    _configured = True
    return log_path
