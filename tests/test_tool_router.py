"""Unit tests for the deterministic per-turn tool-scoper (brain/tool_router)."""

from __future__ import annotations

from glados.brain.tool_router import ServerScope, ToolRouter
from glados.core.adapters import ToolSpec


def _spec(server: str, name: str) -> ToolSpec:
    return ToolSpec(
        server=server, name=name, description=f"{server}.{name}",
        parameters={"type": "object", "properties": {}},
    )


_ALL = [
    _spec("time", "now"),
    _spec("dunnes", "add_to_cart"),
    _spec("dunnes", "view_cart"),
    _spec("weather", "get"),
]


def _router() -> ToolRouter:
    return ToolRouter(scopes={
        "time": ServerScope("time", core=True),
        "dunnes": ServerScope("dunnes", intent_keywords=("cart", "shop", "buy")),
        "weather": ServerScope("weather", intent_keywords=("weather", "forecast")),
        # note: a server present in _ALL but NOT in scopes is "unscoped".
    })


def _servers(specs: list[ToolSpec]) -> set[str]:
    return {s.server for s in specs}


def test_core_server_always_in_scope() -> None:
    out = _router().scope_for("add milk to my cart", _ALL)
    assert "time" in _servers(out)


def test_matched_server_is_scoped_in() -> None:
    out = _router().scope_for("add milk to my cart", _ALL)
    assert "dunnes" in _servers(out)
    assert "weather" not in _servers(out)  # no weather keyword


def test_unmatched_annotated_server_is_dropped() -> None:
    out = _router().scope_for("what is the weather", _ALL)
    assert "weather" in _servers(out)
    assert "dunnes" not in _servers(out)  # cart/shop/buy absent


def test_no_keyword_match_leaves_only_core_and_unscoped() -> None:
    # "tell me a joke" matches no annotated server -> only core (time) survives;
    # there is no unscoped server in _ALL here.
    out = _router().scope_for("tell me a joke", _ALL)
    assert _servers(out) == {"time"}


def test_unscoped_server_always_offered() -> None:
    # A server absent from scopes (here: a toy server) is unscoped -> always in.
    specs = _ALL + [_spec("toy", "echo")]
    out = _router().scope_for("what is the weather", specs)
    assert "toy" in _servers(out)


def test_keyword_is_whole_word() -> None:
    # "buyer" must not match the "buy" keyword (whole-word boundary).
    out = _router().scope_for("who is the buyer", _ALL)
    assert "dunnes" not in _servers(out)


def test_never_returns_whole_registry_on_miss() -> None:
    out = _router().scope_for("nonsense xyzzy", _ALL)
    # dunnes + weather (annotated, unmatched) are excluded; only core remains.
    assert "dunnes" not in _servers(out) and "weather" not in _servers(out)
