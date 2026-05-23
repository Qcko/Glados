"""Tests for the SecretsStore abstraction and bootstrap CLI."""

from __future__ import annotations

import io

import pytest

from glados.core.secrets import InMemorySecrets, KeyringSecrets
from glados.secrets.__main__ import run as cli_run


def test_in_memory_round_trip() -> None:
    s = InMemorySecrets()
    assert s.get("client-tokens", "desk-ui") is None
    s.set("client-tokens", "desk-ui", "abc")
    assert s.get("client-tokens", "desk-ui") == "abc"


def test_in_memory_scope_isolation() -> None:
    s = InMemorySecrets()
    s.set("client-tokens", "desk-ui", "one")
    s.set("dunnes", "desk-ui", "two")
    assert s.get("client-tokens", "desk-ui") == "one"
    assert s.get("dunnes", "desk-ui") == "two"


def test_in_memory_delete_returns_false_when_missing() -> None:
    s = InMemorySecrets()
    assert s.delete("client-tokens", "nope") is False
    s.set("client-tokens", "desk-ui", "x")
    assert s.delete("client-tokens", "desk-ui") is True
    assert s.get("client-tokens", "desk-ui") is None


def test_keyring_secrets_delegates_to_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """KeyringSecrets must hit the real `keyring` module — verify by
    monkeypatching its API and confirming the calls land with the
    namespaced service name."""
    calls: list[tuple] = []

    def fake_set(service, name, value):
        calls.append(("set", service, name, value))

    def fake_get(service, name):
        calls.append(("get", service, name))
        return "stored-value"

    def fake_delete(service, name):
        calls.append(("delete", service, name))

    monkeypatch.setattr("glados.core.secrets.keyring.set_password", fake_set)
    monkeypatch.setattr("glados.core.secrets.keyring.get_password", fake_get)
    monkeypatch.setattr("glados.core.secrets.keyring.delete_password", fake_delete)

    s = KeyringSecrets()
    s.set("client-tokens", "desk-ui", "tok")
    assert s.get("client-tokens", "desk-ui") == "stored-value"
    assert s.delete("client-tokens", "desk-ui") is True

    assert calls == [
        ("set", "glados.client-tokens", "desk-ui", "tok"),
        ("get", "glados.client-tokens", "desk-ui"),
        ("delete", "glados.client-tokens", "desk-ui"),
    ]


# ---- CLI ---------------------------------------------------------------


def test_cli_set_then_get_round_trip() -> None:
    store = InMemorySecrets()
    out = io.StringIO()
    err = io.StringIO()
    rc = cli_run(
        ["set", "client-tokens", "desk-ui"],
        store=store,
        prompt=lambda _: "secret-value",
        out=out,
        err=err,
    )
    assert rc == 0
    assert store.get("client-tokens", "desk-ui") == "secret-value"

    out2 = io.StringIO()
    rc2 = cli_run(
        ["get", "client-tokens", "desk-ui"], store=store, out=out2, err=err
    )
    assert rc2 == 0
    assert out2.getvalue().strip() == "secret-value"


def test_cli_set_rejects_empty_value() -> None:
    store = InMemorySecrets()
    err = io.StringIO()
    rc = cli_run(
        ["set", "client-tokens", "desk-ui"],
        store=store,
        prompt=lambda _: "",
        err=err,
    )
    assert rc == 2
    assert store.get("client-tokens", "desk-ui") is None


def test_cli_get_returns_1_when_missing() -> None:
    store = InMemorySecrets()
    err = io.StringIO()
    rc = cli_run(["get", "client-tokens", "absent"], store=store, err=err)
    assert rc == 1


def test_cli_delete_when_missing_returns_1() -> None:
    store = InMemorySecrets()
    err = io.StringIO()
    rc = cli_run(["delete", "client-tokens", "absent"], store=store, err=err)
    assert rc == 1


def test_cli_delete_when_present_returns_0() -> None:
    store = InMemorySecrets({("client-tokens", "desk-ui"): "x"})
    out = io.StringIO()
    err = io.StringIO()
    rc = cli_run(
        ["delete", "client-tokens", "desk-ui"], store=store, out=out, err=err
    )
    assert rc == 0
    assert store.get("client-tokens", "desk-ui") is None
