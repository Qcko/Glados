from __future__ import annotations

import pytest

from glados.mcp.registry import CallEnvelope, MCPRegistry
from glados.servers.toy_server import TOY_TOOLS, AddTool, EchoTool, RollDiceTool


def _env() -> CallEnvelope:
    return CallEnvelope(session_id="s", room_id="desk", speaker_id="qcko")


@pytest.mark.asyncio
async def test_echo_returns_text() -> None:
    res = await EchoTool().call({"text": "hi"}, _env())
    assert res.ok and res.content == {"text": "hi"}


@pytest.mark.asyncio
async def test_echo_rejects_non_string() -> None:
    res = await EchoTool().call({"text": 7}, _env())
    assert not res.ok and res.error is not None and "string" in res.error


@pytest.mark.asyncio
async def test_add_sums_numbers() -> None:
    res = await AddTool().call({"a": 2, "b": 3.5}, _env())
    assert res.ok and res.content == {"sum": 5.5}


@pytest.mark.asyncio
async def test_add_rejects_non_number() -> None:
    res = await AddTool().call({"a": "x", "b": 1}, _env())
    assert not res.ok


@pytest.mark.asyncio
async def test_roll_dice_default_count() -> None:
    res = await RollDiceTool().call({"sides": 6}, _env())
    assert res.ok
    rolls = res.content["rolls"]
    assert len(rolls) == 1 and 1 <= rolls[0] <= 6
    assert res.content["total"] == sum(rolls)


@pytest.mark.asyncio
async def test_roll_dice_count_and_bounds() -> None:
    res = await RollDiceTool().call({"sides": 20, "count": 5}, _env())
    assert res.ok
    rolls = res.content["rolls"]
    assert len(rolls) == 5 and all(1 <= r <= 20 for r in rolls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"sides": 1},
        {"sides": 1001},
        {"sides": 6, "count": 0},
        {"sides": 6, "count": 101},
        {"sides": True},
        {"sides": 6, "count": True},
    ],
)
async def test_roll_dice_rejects_bad_args(args: dict) -> None:
    res = await RollDiceTool().call(args, _env())
    assert not res.ok


@pytest.mark.asyncio
async def test_add_rejects_bool() -> None:
    res = await AddTool().call({"a": True, "b": 1}, _env())
    assert not res.ok


@pytest.mark.asyncio
async def test_toy_tools_dispatch_through_registry() -> None:
    reg = MCPRegistry()
    for tool in TOY_TOOLS:
        reg.register(tool)
    qualified = {s.qualified for s in reg.specs()}
    assert {"toy.echo", "toy.add", "toy.roll_dice"} <= qualified

    res = await reg.dispatch("toy", "add", {"a": 1, "b": 2}, _env())
    assert res.ok and res.content == {"sum": 3}
