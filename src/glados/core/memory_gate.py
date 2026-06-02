"""LocalGuard-for-prompts integration seam (ARCH §14).

GLaDOS owns only the *integration*: at memory-load it asks LocalGuard whether
a server-shipped lessons blob exactly matches a human-approved baseline,
**fails closed**, and guard-wraps approved content as untrusted reference
data before it touches the system prompt. LocalGuard owns the verdict, the
baseline store, the injection-scan ruleset, and the approve-with-diff flow.

The boundary is LocalGuard's CLI (`localguard memory check`), not a Python
import: the two repos keep separate venvs and the baseline store lives under
LocalGuard's own `LOCALGUARD_LIBRARY` root. A pure hash lookup, deterministic,
no model — the runtime path has no injection surface of its own.

Fail-closed is the whole point. Every non-APPROVED outcome — unknown content,
changed content, LocalGuard absent, CLI error, malformed output — yields None,
and None means *nothing is injected*. New or edited lessons must be approved
out-of-band (`localguard memory approve`) before they reach a prompt again.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess

log = logging.getLogger(__name__)

# Override to point at a non-PATH binary or a wrapper, e.g.
# `GLADOS_LOCALGUARD_CMD="uv run --project E:/dev/localguard localguard"`.
# Shell-split, so multi-word wrappers work; never run through a shell.
_LOCALGUARD_CMD_ENV = "GLADOS_LOCALGUARD_CMD"
_DEFAULT_LOCALGUARD_CMD = "localguard"

# Time budget for the (cheap, deterministic) hash lookup. A hung or slow
# LocalGuard must not stall GLaDOS startup; the timeout fails closed.
_CHECK_TIMEOUT_S = 10.0


def vet(source: str, blob: str) -> str | None:
    """Return guard-wrapped memory ready for prompt injection, or None.

    `source` is the stable origin key LocalGuard baselines against — the MCP
    server id. None means "do not inject" for any reason (not approved, or
    LocalGuard unreachable); the caller injects nothing.
    """
    if not _is_approved(source, blob):
        return None
    return _wrap(source, blob)


def _is_approved(source: str, blob: str) -> bool:
    argv = _localguard_cmd()
    if not argv:
        # A blank or unsplittable GLADOS_LOCALGUARD_CMD. Fail closed rather
        # than IndexError on cmd[0] (which would fault startup open-loud).
        log.warning(
            "memory gate: %s split to no command; injecting no memory for %r",
            _LOCALGUARD_CMD_ENV,
            source,
        )
        return False
    cmd = [*argv, "memory", "check", "-", "--source", source, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            input=blob,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_CHECK_TIMEOUT_S,
        )
    except FileNotFoundError:
        # LocalGuard not installed / not on PATH. Fail closed: a missing
        # gate must never become an open door.
        log.warning(
            "memory gate: localguard not found (%r); injecting no memory for %r. "
            "Set %s or install localguard.",
            cmd[0],
            source,
            _LOCALGUARD_CMD_ENV,
        )
        return False
    except Exception as e:  # noqa: BLE001
        # TimeoutExpired, OSError, ValueError (bad args), or anything else.
        # The gate's only acceptable failure mode is "inject nothing" — never
        # let a check error crash the caller (the server lifespan) open.
        log.warning("memory gate: localguard check failed for %r: %s", source, e)
        return False
    if proc.returncode == 0:
        log.info("memory gate: APPROVED lessons for %r", source)
        return True
    # Non-zero is LocalGuard's fail-closed signal (unknown/changed/blocked).
    # stderr carries the human reason; surface it so an operator knows to run
    # `localguard memory approve` on the new content.
    reason = (proc.stdout or proc.stderr or "").strip().splitlines()
    log.info(
        "memory gate: BLOCKED lessons for %r (exit %d): %s",
        source,
        proc.returncode,
        reason[-1] if reason else "no detail",
    )
    return False


def _localguard_cmd() -> list[str]:
    raw = os.environ.get(_LOCALGUARD_CMD_ENV, "").strip()
    if not raw:
        return [_DEFAULT_LOCALGUARD_CMD]
    if os.name != "nt":
        return shlex.split(raw)
    # On Windows, POSIX-mode shlex would eat the backslashes in a path like
    # `E:\dev\localguard\...`. posix=False keeps them but leaves surrounding
    # quotes in each token, so strip a matching pair back off afterward.
    return [_strip_quotes(tok) for tok in shlex.split(raw, posix=False)]


def _strip_quotes(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
        return tok[1:-1]
    return tok


def _wrap(source: str, blob: str) -> str:
    """Frame approved memory as untrusted reference data (ARCH §14 layer 4).

    Even hash-approved content is delimited and labelled, never spliced in as
    raw system instructions — the §7 `<external>` discipline applied to
    memory. Any literal closing tag inside the blob is defanged so the blob
    cannot break out of its own delimiter (belt-and-braces; LocalGuard's
    approval scan also flags framing-tag defang attempts).
    """
    safe = _TAG_LIKE.sub(lambda m: "&lt;" + m.group(0)[1:], blob)
    # `source` is the operator-set server id, but defang `"`/`<`/`>` so a
    # stray char can't malform the opening tag's attribute.
    src = re.sub(r'["<>]', "", source)
    return f'<memory-notes source="{src}">\n{safe}\n</memory-notes>'


# Any token that looks like an opening or closing <memory-notes> tag,
# tolerant of case, internal whitespace, and attributes — so a blob can't
# break out of (or forge) its own delimiter. We neutralise only the leading
# `<`; LocalGuard's approval-time scan is the primary defence (ARCH §14 L3),
# this is belt-and-braces matching the <external> discipline.
_TAG_LIKE = re.compile(r"<\s*/?\s*memory-notes", re.IGNORECASE)
