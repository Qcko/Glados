"""Secrets store abstraction backed by the OS keyring.

Per ARCH section 9: TOML holds handles only, never values. Real values live in
the OS-native credential store (Windows Credential Manager, libsecret,
macOS Keychain) under service names like `glados.client-tokens`,
`glados.dunnes`, `glados.spotify`. The scope is the second half of the
service name; the name is the keyring username.

Tests use `InMemorySecrets` so they never touch the OS keyring.
"""

from __future__ import annotations

from typing import Protocol

import jaraco.context
import keyring
from keyring.errors import PasswordDeleteError


def _block_jaraco_tarball() -> None:
    """Defense-in-depth: neutralise `jaraco.context.tarball`.

    GLaDOS never extracts tar archives via jaraco.context -- `keyring` only
    uses its decorator helpers (`@suppress`, `@on_interrupt`, etc.). The
    `tarball` helper has had a zip-slip path-traversal CVE in the past
    (CVE-2026-23949, fixed in 6.1.0; we ship >=6.1.2 so we're not exposed
    today). Stubbing it out at import time means any future regression or
    new caller that tries to use it fails loud instead of silently
    extracting attacker-controlled paths.
    """

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(
            "jaraco.context.tarball is disabled in GLaDOS (see core/secrets.py)"
        )

    jaraco.context.tarball = _blocked  # type: ignore[attr-defined]


_block_jaraco_tarball()


_SERVICE_PREFIX = "glados"


def _service(scope: str) -> str:
    return f"{_SERVICE_PREFIX}.{scope}"


class SecretsStore(Protocol):
    def get(self, scope: str, name: str) -> str | None: ...
    def set(self, scope: str, name: str, value: str) -> None: ...
    # Returns True if a value was deleted, False if no entry existed.
    # Backends that can't distinguish "missing" from "delete failed" must
    # err on the side of False so callers don't over-report success.
    def delete(self, scope: str, name: str) -> bool: ...


class KeyringSecrets:
    """Real OS-keyring impl. Service names namespaced under `glados.<scope>`."""

    def get(self, scope: str, name: str) -> str | None:
        return keyring.get_password(_service(scope), name)

    def set(self, scope: str, name: str, value: str) -> None:
        keyring.set_password(_service(scope), name, value)

    def delete(self, scope: str, name: str) -> bool:
        try:
            keyring.delete_password(_service(scope), name)
            return True
        except PasswordDeleteError:
            return False


class InMemorySecrets:
    """Dict-backed impl for tests. Never touches the OS keyring."""

    def __init__(self, initial: dict[tuple[str, str], str] | None = None) -> None:
        self._store: dict[tuple[str, str], str] = dict(initial or {})

    def get(self, scope: str, name: str) -> str | None:
        return self._store.get((scope, name))

    def set(self, scope: str, name: str, value: str) -> None:
        self._store[(scope, name)] = value

    def delete(self, scope: str, name: str) -> bool:
        return self._store.pop((scope, name), None) is not None
