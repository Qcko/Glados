"""Which release this process is running, for the deploy's readiness check.

A deploy that only polls "is something answering" passes against a stale
process that never stopped. Reporting the commit the process booted from lets
the deploy compare it with the tag it just installed. Read once at boot:
the checkout can move under a running process, and what matters is what was
loaded, not what is on disk now.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_GIT_TIMEOUT_S = 5.0


def describe_release(repo_root: Path) -> dict[str, str | None]:
    """`{"sha": <HEAD sha>, "tag": <tag exactly at HEAD>}`; either is None when
    git cannot say (no checkout, no git, or HEAD is not on a tag)."""
    return {
        "sha": _git(repo_root, "rev-parse", "HEAD"),
        "tag": _git(repo_root, "describe", "--tags", "--exact-match", "HEAD"),
    }


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603,S607 -- fixed argv, no shell
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("release: git %s unavailable: %s", args[0], exc)
        return None
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None
