"""A MUTATING call recovered from TEXT is confirmed, even when un-gated.

`dunnes.add_to_cart_by_name` ships `requires_confirmation = false` on purpose:
asking "may I add milk?" after the user said "add milk" is bad voice UX. That
reasoning holds for a call the model handed over as STRUCTURE. It does not hold
for one parsed out of assistant text, because text is the channel `<external>`
content arrives on -- so provenance, not just the tool, decides the gate
(ARCH section 7).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel

from glados.core.adapters import LLMText, LLMToolCall, ToolSpec
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.protocols import ToolConfirmResponse
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry


class _UngatedMutatingTool:
    """Mutating, but deliberately NOT requires_confirmation -- the shape the
    real cart tools ship with."""

    spec = ToolSpec(
        server="shop",
        name="add_to_cart",
        description="add an item to the cart",
        parameters={"type": "object"},
        requires_confirmation=False,
        mutating=True,
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(args)
        return MCPCallResult(ok=True, content={"added": True})


class _OneCallLLM:
    def __init__(self, *, from_text: bool) -> None:
        self._from_text = from_text
        self._n = 0

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield LLMToolCall(
                call_id="c1", server="shop", name="add_to_cart",
                args={"q": "milk"}, from_text=self._from_text,
            )
        else:
            yield LLMText(text="done")


@asynccontextmanager
async def _make_org(tmp: Path, *, from_text: bool):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    bindings = [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="u")]
    by_id = {b.client_id: b for b in bindings}
    mcp = MCPRegistry()
    tool = _UngatedMutatingTool()
    mcp.register(tool)
    org = Organizer(
        llm=_OneCallLLM(from_text=from_text),
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
        confirm_timeout_s=30.0,
    )
    try:
        yield org, sink, tool
    finally:
        await org.close()


def _confirm_requests(sink: list) -> list[dict]:
    return [m for _, m in sink if m.get("type") == "tool_confirm_request"]


async def _await_confirm(sink: list, timeout_s: float = 2.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if _confirm_requests(sink):
            return _confirm_requests(sink)[0]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no tool_confirm_request after {timeout_s}s; sink={sink}")


async def test_structured_mutating_call_is_not_gated(tmp_path: Path) -> None:
    # The existing, deliberate behaviour: no prompt for a normal cart add.
    async with _make_org(tmp_path, from_text=False) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()
        assert tool.calls == [{"q": "milk"}]
        assert _confirm_requests(sink) == []


async def test_text_parsed_mutating_call_is_gated(tmp_path: Path) -> None:
    async with _make_org(tmp_path, from_text=True) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "add milk")
        req = await _await_confirm(sink)
        await org.handle_tool_confirm_response(
            "desk-ui", ToolConfirmResponse(request_id=req["request_id"], granted=True)
        )
        await org.flush()
        assert tool.calls == [{"q": "milk"}]


async def test_text_parsed_mutating_call_is_skipped_when_denied(tmp_path: Path) -> None:
    async with _make_org(tmp_path, from_text=True) as (org, sink, tool):
        await org.handle_user_text("desk-ui", "add milk")
        req = await _await_confirm(sink)
        await org.handle_tool_confirm_response(
            "desk-ui", ToolConfirmResponse(request_id=req["request_id"], granted=False)
        )
        await org.flush()
        assert tool.calls == []
