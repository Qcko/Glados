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

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Override to point at a non-PATH binary or a wrapper, e.g.
# `GLADOS_LOCALGUARD_CMD="uv run --project E:/dev/localguard localguard"`.
# Shell-split, so multi-word wrappers work; never run through a shell.
_LOCALGUARD_CMD_ENV = "GLADOS_LOCALGUARD_CMD"
_DEFAULT_LOCALGUARD_CMD = "localguard"

# Time budget for the (cheap, deterministic) hash lookup. A hung or slow
# LocalGuard must not stall GLaDOS startup; the timeout fails closed.
_CHECK_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GateResult:
    """Outcome of vetting one server's lessons blob.

    `note` is the guard-wrapped, inject-ready string and is present iff
    `approved`. `reason` is LocalGuard's human-readable block cause and is
    present iff blocked. `sha256`/`length` are metadata-safe descriptors of
    the blob (a hex digest and character count) — they carry **none of the
    untrusted bytes**, so they are safe to surface to an operator on any
    channel (ARCH §14 BLOCK-notice surface).
    """

    source: str
    approved: bool
    sha256: str
    length: int
    note: str | None = None
    reason: str | None = None


def check(source: str, blob: str) -> GateResult:
    """Vet a server's lessons blob through LocalGuard, fail-closed.

    `source` is the stable origin key LocalGuard baselines against — the MCP
    server id. On approval the result carries an inject-ready `note`; on any
    non-approved outcome (unknown/changed content, LocalGuard unreachable, CLI
    error) it carries a `reason` and `note is None`, so the caller injects
    nothing but can still surface the BLOCK as metadata.
    """
    sha256 = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    approved, reason = _check_approval(source, blob)
    if approved:
        return GateResult(source, True, sha256, len(blob), note=_wrap(source, blob))
    return GateResult(source, False, sha256, len(blob), reason=reason)


def vet(source: str, blob: str) -> str | None:
    """Guard-wrapped memory ready for prompt injection, or None.

    Thin convenience over `check` for callers that only need the inject path;
    None means "do not inject" for any reason.
    """
    return check(source, blob).note


def _check_approval(source: str, blob: str) -> tuple[bool, str | None]:
    """Return (approved, reason). `reason` is None when approved, else a
    short, metadata-only block cause (never echoes the blob's bytes)."""
    argv = _localguard_cmd()
    if not argv:
        # A blank or unsplittable GLADOS_LOCALGUARD_CMD. Fail closed rather
        # than IndexError on cmd[0] (which would fault startup open-loud).
        log.warning(
            "memory gate: %s split to no command; injecting no memory for %r",
            _LOCALGUARD_CMD_ENV,
            source,
        )
        return False, f"{_LOCALGUARD_CMD_ENV} is set but split to no command"
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
        return False, "localguard not found — gate unavailable"
    except Exception as e:  # noqa: BLE001
        # TimeoutExpired, OSError, ValueError (bad args), or anything else.
        # The gate's only acceptable failure mode is "inject nothing" — never
        # let a check error crash the caller (the server lifespan) open.
        log.warning("memory gate: localguard check failed for %r: %s", source, e)
        return False, f"localguard check failed ({type(e).__name__})"
    if proc.returncode == 0:
        log.info("memory gate: APPROVED lessons for %r", source)
        return True, None
    # Non-zero is LocalGuard's fail-closed signal (unknown/changed/blocked).
    # Log LocalGuard's own detail line locally — the log is operator-local and
    # not the metadata channel — but the *returned* reason rides the operator
    # metadata channel (UI push + /admin/memory), where §14 promises no
    # untrusted bytes. So we never forward LocalGuard's free-text `reason`; we
    # forward only its closed-vocabulary `reason_code` (and only after a
    # syntactic safety check), falling back to the exit code. The operator
    # reads the actual bytes later, inert, in the review pane — never here.
    detail = (proc.stdout or proc.stderr or "").strip().splitlines()
    log.info(
        "memory gate: BLOCKED lessons for %r (exit %d): %s",
        source,
        proc.returncode,
        detail[-1] if detail else "no detail",
    )
    code = _safe_reason_code(proc.stdout)
    cause = code if code is not None else f"exit {proc.returncode}"
    return False, (
        f"not approved ({cause}); run "
        "`localguard memory approve` on the new content"
    )


# A LocalGuard `reason_code` is a closed-vocabulary categorisation of why a
# blob was blocked (e.g. "unknown_content", "baseline_not_approved",
# "scan_rule:role_tag_injection"). We forward it to the operator metadata
# channel only after a *syntactic* safety check: a lowercase token of bounded
# length, colon-namespaced at most, no whitespace/quotes/punctuation. A blob
# fragment cannot satisfy this shape, so even a regressed or hostile LocalGuard
# cannot smuggle untrusted bytes through this field — GLaDOS upholds the §14
# no-echo invariant at its own boundary rather than trusting the producer.
# `\Z` (not `$`) so a trailing newline can't ride through: `$` matches just
# before a final `\n`, which would let "foo\n" pass and forward the newline.
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_:]{0,63}\Z")


def _safe_reason_code(stdout: str | None) -> str | None:
    """Extract LocalGuard's `reason_code` from a `--json` verdict, or None.

    None when stdout is absent, not JSON, not an object, lacks the field, or
    the value fails the syntactic safety check — in every such case the caller
    falls back to the exit-code reason. Forward-compatible: returns None today
    (LocalGuard does not yet emit the field) without any error.
    """
    if not stdout:
        return None
    try:
        verdict = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(verdict, dict):
        return None
    code = verdict.get("reason_code")
    if isinstance(code, str) and _REASON_CODE_RE.fullmatch(code):
        return code
    return None


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
