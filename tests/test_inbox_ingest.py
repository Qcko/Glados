"""Inbox ingest: client tokens move from files into the keyring (DESIGN.md,
"registering client tokens without the service password")."""

from __future__ import annotations

import io
import json
import os

import pytest

from glados.core.secrets import InMemorySecrets
from glados.secrets import run
from glados.secrets.ingest import ADMIN_FILE, ADMIN_NAME, ADMIN_SCOPE, SCOPE, ingest_dir

TOKEN = "A" * 20 + "b-_" + "9" * 20
NON_ASCII = chr(0xE9) * 40


def _drop(inbox, client_id, value=TOKEN + "\r\n"):
    path = inbox / f"{client_id}.token"
    path.write_bytes(value.encode("utf-8"))
    return path


def test_allowlisted_token_is_stored_and_file_deleted(tmp_path):
    store = InMemorySecrets()
    path = _drop(tmp_path, "kitchen-mic")

    outcomes = ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert store.get(SCOPE, "kitchen-mic") == TOKEN
    assert not path.exists()
    assert [(o.name, o.result, o.reason) for o in outcomes] == [("kitchen-mic", "registered", "")]


def test_existing_token_is_reported_as_rotated(tmp_path):
    store = InMemorySecrets({(SCOPE, "kitchen-mic"): "old"})
    _drop(tmp_path, "kitchen-mic")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert outcome.result == "rotated"
    assert store.get(SCOPE, "kitchen-mic") == TOKEN


def test_unknown_client_is_kept_and_not_stored(tmp_path):
    store = InMemorySecrets()
    path = _drop(tmp_path, "stranger")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert outcome.result == "kept"
    assert path.exists()
    assert store.get(SCOPE, "stranger") is None


@pytest.mark.parametrize(
    "value",
    ["", "\n", "short", TOKEN + " ", TOKEN + "\n\n", "x" * 200, NON_ASCII, TOKEN + "\nmore"],
)
def test_malformed_value_is_kept(tmp_path, value):
    store = InMemorySecrets()
    path = _drop(tmp_path, "kitchen-mic", value)

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert (outcome.result, outcome.reason) == ("kept", "value is not a token")
    assert path.exists()
    assert store.get(SCOPE, "kitchen-mic") is None


def test_utf8_bom_from_windows_powershell_is_tolerated(tmp_path):
    store = InMemorySecrets()
    (tmp_path / "kitchen-mic.token").write_bytes(b"\xef\xbb\xbf" + TOKEN.encode() + b"\r\n")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert outcome.result == "registered"
    assert store.get(SCOPE, "kitchen-mic") == TOKEN


def test_unreadable_file_is_kept_and_the_rest_still_ingest(tmp_path, monkeypatch):
    store = InMemorySecrets()
    _drop(tmp_path, "kitchen-mic")
    _drop(tmp_path, "kitchen-speaker")
    real = os.lstat

    def flaky(path, *a, **k):
        if str(path).endswith("kitchen-mic.token"):
            raise PermissionError("in use")
        return real(path, *a, **k)

    monkeypatch.setattr(os, "lstat", flaky)

    outcomes = ingest_dir(tmp_path, ["kitchen-mic", "kitchen-speaker"], store)

    assert [(o.name, o.result, o.reason) for o in outcomes] == [
        ("kitchen-mic", "kept", "read error: PermissionError"),
        ("kitchen-speaker", "registered", ""),
    ]


def test_partial_upload_and_other_files_are_ignored(tmp_path):
    (tmp_path / "kitchen-mic.token.part").write_text(TOKEN)
    (tmp_path / "Kitchen-Mic.token").write_text(TOKEN)
    (tmp_path / "notes.txt").write_text(TOKEN)

    assert ingest_dir(tmp_path, ["kitchen-mic"], InMemorySecrets()) == []


def test_hard_linked_file_is_refused(tmp_path):
    path = _drop(tmp_path, "kitchen-mic")
    os.link(path, tmp_path / "elsewhere")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], InMemorySecrets())

    assert outcome.reason == "file has more than one hard link"


def test_symlink_is_refused(tmp_path):
    target = tmp_path / "target"
    target.write_text(TOKEN)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    try:
        os.symlink(target, inbox / "kitchen-mic.token")
    except OSError:
        pytest.skip("creating symlinks needs a privilege this account lacks")

    (outcome,) = ingest_dir(inbox, ["kitchen-mic"], InMemorySecrets())

    assert outcome.reason == "file is a link or reparse point"
    assert target.exists()


class _LyingStore(InMemorySecrets):
    def get(self, scope, name):
        return "something else" if (scope, name) in self._store else None


def test_read_back_mismatch_keeps_file(tmp_path):
    path = _drop(tmp_path, "kitchen-mic")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], _LyingStore())

    assert (outcome.result, outcome.reason) == ("kept", "read-back mismatch")
    assert path.exists()


class _BrokenStore(InMemorySecrets):
    def set(self, scope, name, value):
        raise RuntimeError(value)


def test_keyring_error_keeps_file_and_never_echoes_the_value(tmp_path):
    path = _drop(tmp_path, "kitchen-mic")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], _BrokenStore())

    assert outcome.result == "kept"
    assert TOKEN not in outcome.reason
    assert path.exists()


def test_failed_delete_after_store_warns(tmp_path):
    store = InMemorySecrets()
    _drop(tmp_path, "kitchen-mic")

    def refuse(_path):
        raise PermissionError("denied")

    (outcome,) = ingest_dir(tmp_path, ["kitchen-mic"], store, remove=refuse)

    assert outcome.result == "registered"
    assert outcome.reason.startswith("WARN")
    assert store.get(SCOPE, "kitchen-mic") == TOKEN


def test_rerun_after_success_is_a_no_op(tmp_path):
    store = InMemorySecrets()
    _drop(tmp_path, "kitchen-mic")
    ingest_dir(tmp_path, ["kitchen-mic"], store)

    assert ingest_dir(tmp_path, ["kitchen-mic"], store) == []


def _drop_admin(inbox, value=TOKEN + "\r\n"):
    path = inbox / ADMIN_FILE
    path.write_bytes(value.encode("utf-8"))
    return path


def test_admin_secret_is_stored_without_being_a_listed_client(tmp_path):
    store = InMemorySecrets()
    path = _drop_admin(tmp_path)

    (outcome,) = ingest_dir(tmp_path, [], store)

    assert (outcome.name, outcome.result, outcome.reason) == ("admin/observe-token", "registered", "")
    assert store.get(ADMIN_SCOPE, ADMIN_NAME) == TOKEN
    assert store.get(SCOPE, "admin.observe-token") is None
    assert not path.exists()


def test_admin_secret_still_gets_the_value_check(tmp_path):
    store = InMemorySecrets()
    path = _drop_admin(tmp_path, "short")

    (outcome,) = ingest_dir(tmp_path, [], store)

    assert (outcome.result, outcome.reason) == ("kept", "value is not a token")
    assert path.exists()
    assert store.get(ADMIN_SCOPE, ADMIN_NAME) is None


def test_admin_secret_still_gets_the_link_check(tmp_path):
    path = _drop_admin(tmp_path)
    os.link(path, tmp_path / "elsewhere")

    (outcome,) = ingest_dir(tmp_path, [], InMemorySecrets())

    assert outcome.reason == "file has more than one hard link"


def test_a_client_named_admin_stays_a_client_token(tmp_path):
    store = InMemorySecrets()
    _drop(tmp_path, "admin")

    (outcome,) = ingest_dir(tmp_path, ["admin"], store)

    assert outcome.name == "admin"
    assert store.get(SCOPE, "admin") == TOKEN
    assert store.get(ADMIN_SCOPE, ADMIN_NAME) is None


def test_other_dotted_names_are_ignored(tmp_path):
    (tmp_path / "admin.other.token").write_text(TOKEN)
    (tmp_path / "client-tokens.kitchen-mic.token").write_text(TOKEN)

    assert ingest_dir(tmp_path, ["kitchen-mic"], InMemorySecrets()) == []


def test_cli_uses_merged_config_and_writes_summary_without_values(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    local = tmp_path / "local"
    inbox = tmp_path / "inbox"
    for d in (configs, local, inbox):
        d.mkdir()
    (configs / "glados.toml").write_text('[auth]\nclients = ["desk-ui"]\n')
    (local / "glados.local.toml").write_text('[auth]\nclients = ["kitchen-mic"]\n')
    monkeypatch.setenv("GLADOS_CONFIG_DIR", str(configs))
    monkeypatch.setenv("GLADOS_LOCAL_CONFIG_DIR", str(local))
    _drop(inbox, "kitchen-mic")
    _drop(inbox, "stranger")
    summary = tmp_path / "summary.json"
    store = InMemorySecrets()
    out = io.StringIO()

    code = run(["ingest", str(inbox), "--summary", str(summary)], store=store, out=out)

    assert code == 1
    assert store.get(SCOPE, "kitchen-mic") == TOKEN
    text = summary.read_text(encoding="utf-8") + out.getvalue()
    assert TOKEN not in text
    assert {o["name"]: o["result"] for o in json.loads(summary.read_text())} == {
        "kitchen-mic": "registered",
        "stranger": "kept",
    }
