"""Move client tokens (and the one reserved admin secret) from an inbox
directory into the running account's keyring.

Run by the prod launcher as the service account, so the token lands in the
vault the server reads without anyone typing that account's password (see
DESIGN.md, "registering client tokens without the service password"). The
inbox is a transfer area, never a store: a token leaves it only once the
keyring holds it and reads it back. Outcomes name clients, never values.
"""

from __future__ import annotations

import hmac
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..core.secrets import SecretsStore

SCOPE = "client-tokens"
ADMIN_FILE = "admin.observe-token.token"
ADMIN_SCOPE = "admin"
ADMIN_NAME = "observe-token"
_NAME = re.compile(r"[a-z0-9-]+\.token")
_VALUE = re.compile(r"[A-Za-z0-9_-]{32,128}")
_UTF8_BOM = b"\xef\xbb\xbf"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class _Target:
    label: str
    scope: str
    name: str
    refusal: str


@dataclass(frozen=True)
class Outcome:
    name: str
    result: str
    reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def ingest_dir(
    inbox: Path,
    allowed: Iterable[str],
    store: SecretsStore,
    *,
    scope: str = SCOPE,
    remove: Callable[[Path], None] = os.remove,
) -> list[Outcome]:
    allowed_ids = set(allowed)
    return [
        _ingest_file(path, target, store, remove)
        for path in sorted(inbox.iterdir())
        if (target := _target_for(path.name, allowed_ids, scope)) is not None
    ]


def _target_for(filename: str, allowed_ids: set[str], scope: str) -> _Target | None:
    if filename == ADMIN_FILE:
        return _Target(f"{ADMIN_SCOPE}/{ADMIN_NAME}", ADMIN_SCOPE, ADMIN_NAME, "")
    if not _NAME.fullmatch(filename):
        return None
    client_id = filename[: -len(".token")]
    refusal = "" if client_id in allowed_ids else "client id is not in [auth].clients"
    return _Target(client_id, scope, client_id, refusal)


def _ingest_file(
    path: Path,
    target: _Target,
    store: SecretsStore,
    remove: Callable[[Path], None],
) -> Outcome:
    try:
        refusal = target.refusal or _refuse_file(path)
        value = None if refusal else _read_value(path)
    except OSError as exc:
        return Outcome(target.label, "kept", f"read error: {type(exc).__name__}")
    if refusal:
        return Outcome(target.label, "kept", refusal)
    if value is None:
        return Outcome(target.label, "kept", "value is not a token")
    return _store_and_remove(path, target, value, store, remove)


def _refuse_file(path: Path) -> str:
    info = os.lstat(path)
    if _is_link(path, info):
        return "file is a link or reparse point"
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    if info.st_nlink != 1:
        return "file has more than one hard link"
    return ""


def _is_link(path: Path, info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & _REPARSE_POINT)


def _read_value(path: Path) -> str | None:
    raw = path.read_bytes().removeprefix(_UTF8_BOM)
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith(b"\n"):
        raw = raw[:-1]
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    return value if _VALUE.fullmatch(value) else None


def _store_and_remove(
    path: Path,
    target: _Target,
    value: str,
    store: SecretsStore,
    remove: Callable[[Path], None],
) -> Outcome:
    try:
        previous = store.get(target.scope, target.name)
        store.set(target.scope, target.name, value)
        stored = store.get(target.scope, target.name) or ""
    except Exception as exc:  # noqa: BLE001 - any keyring failure keeps the file
        return Outcome(target.label, "kept", f"keyring error: {type(exc).__name__}")
    if not hmac.compare_digest(stored.encode(), value.encode()):
        return Outcome(target.label, "kept", "read-back mismatch")
    result = "registered" if previous is None else "rotated"
    try:
        remove(path)
    except OSError as exc:
        return Outcome(target.label, result, f"WARN stored but file not deleted: {type(exc).__name__}")
    return Outcome(target.label, result)
