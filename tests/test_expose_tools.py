"""expose_tools allowlist (DESIGN-tool-exposure.md, S1)."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from glados.core.adapters import ToolSpec
from glados.core.config import ServerEntry, ServersConfig
from glados.core.server import _expose


def _entry(**kw) -> ServerEntry:
    return ServerEntry(id="calendar", command="x", **kw)


def _specs(*names: str) -> list[ToolSpec]:
    return [ToolSpec(server="calendar", name=n, description="", parameters={}) for n in names]


def test_no_allowlist_keeps_every_listed_tool() -> None:
    assert [s.name for s in _expose(_entry(), _specs("a", "b"))] == ["a", "b"]


def test_allowlist_keeps_only_named_tools_in_server_order() -> None:
    kept = _expose(_entry(expose_tools=["b", "a"]), _specs("a", "grant_access", "b"))
    assert [s.name for s in kept] == ["a", "b"]


def test_a_tool_the_server_adds_later_stays_hidden(caplog) -> None:
    with caplog.at_level(logging.INFO):
        kept = _expose(_entry(expose_tools=["a"]), _specs("a", "new_tool"))
    assert [s.name for s in kept] == ["a"]
    assert "new_tool" in caplog.text


def test_an_allowlist_typo_is_warned_about(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        kept = _expose(_entry(expose_tools=["get_agnda"]), _specs("get_agenda"))
    assert kept == []
    assert "get_agnda" in caplog.text


def test_an_empty_allowlist_exposes_nothing() -> None:
    assert _expose(_entry(expose_tools=[]), _specs("a")) == []


def test_shipped_calendar_entry_hides_the_acl_tools() -> None:
    raw = tomllib.loads(Path("configs/servers.example.toml").read_text(encoding="utf-8"))
    calendar = next(e for e in ServersConfig(**raw).server if e.id == "calendar")
    assert calendar.expose_tools is not None
    for name in ("grant_access", "revoke_access", "list_grants", "delete_board"):
        assert name not in calendar.expose_tools


def test_an_overlay_for_a_hidden_tool_is_warned_about(caplog) -> None:
    entry = _entry(expose_tools=["a"], tool_overlays={"b": {"mutating": True}})
    with caplog.at_level(logging.WARNING):
        _expose(entry, _specs("a", "b"))
    assert "tool_overlays name hidden tools: b" in caplog.text
