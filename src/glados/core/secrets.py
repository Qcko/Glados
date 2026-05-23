"""Secrets store abstraction backed by the OS keyring.

Per ARCH §9: TOML holds handles only, never values. Real values live in
the OS-native credential store (Windows Credential Manager, libsecret,
macOS Keychain) under service names like `glados.client-tokens`,
`glados.dunnes`, `glados.spotify`. The scope is the second half of the
service name; the name is the keyring username.

Tests use `InMemorySecrets` so they never touch the OS keyring.
"""

from __future__ import annotations

from typing import Protocol

import keyring
from keyring.errors import PasswordDeleteError

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
