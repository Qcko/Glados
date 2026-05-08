"""Organizer: the only place sessions, queueing, and egress routing live.

v0 step 2 scope: ingress tagging via `ClientBinding`, single-turn sessions
(no continuation yet), per-session tool-calling loop, egress fan-out by
`room_id`. Dedup, fingerprinting, and per-session FIFO queueing land when
audio arrives in v1.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable

from pydantic import BaseModel

from ..mcp.registry import CallEnvelope, MCPRegistry
from .adapters import LLM, LLMMessage, LLMText, LLMToolCall
from .config import ClientBinding
from .protocols import AssistantDelta, Done, ToolCall, ToolResult, Welcome
from .sessions import SessionRegistry
from .traces import TraceStore

SendFn = Callable[[str, BaseModel], Awaitable[None]]
BindingLookup = Callable[[str], ClientBinding | None]
RoomLookup = Callable[[str], list[str]]

_SYSTEM_PROMPT = (
    "You are GLaDOS, a local home assistant. Use tools when they help. "
    "Be concise."
)
_MAX_TOOL_LOOP = 8


class Organizer:
    def __init__(
        self,
        *,
        llm: LLM,
        mcp: MCPRegistry,
        traces: TraceStore,
        sessions: SessionRegistry,
        send: SendFn,
        binding_for_client: BindingLookup,
        clients_in_room: RoomLookup,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.traces = traces
        self.sessions = sessions
        self.send = send
        self.binding_for_client = binding_for_client
        self.clients_in_room = clients_in_room

    async def handle_user_text(self, client_id: str, text: str) -> None:
        binding = self.binding_for_client(client_id)
        if binding is None:
            return
        session = self.sessions.get_or_open(binding.room_id, binding.default_user)
        envelope = CallEnvelope(
            session_id=session.session_id,
            room_id=session.room_id,
            speaker_id=session.speaker_id,
        )
        trace = self.traces.open(session.session_id)
        try:
            trace.event(
                "turn_start",
                room_id=session.room_id,
                speaker_id=session.speaker_id,
                origin_client=client_id,
            )
            trace.event("user_text", text=text)
            await self._broadcast(session.room_id, Welcome(session_id=session.session_id))

            messages: list[LLMMessage] = [
                LLMMessage(role="system", content=_SYSTEM_PROMPT),
                LLMMessage(role="user", content=text),
            ]
            for _ in range(_MAX_TOOL_LOOP):
                pending_calls, assistant_text = await self._run_one_llm_pass(
                    session.session_id, session.room_id, messages, trace
                )
                if not pending_calls:
                    break
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=assistant_text or None,
                        tool_calls=pending_calls,
                    )
                )
                await self._run_tool_calls(
                    session.session_id, session.room_id, envelope, pending_calls, messages, trace
                )
            else:
                await self._handle_loop_exhausted(session.session_id, session.room_id, trace)
            await self._broadcast(session.room_id, Done(session_id=session.session_id))
            trace.event("done")
        finally:
            trace.close()

    async def _run_one_llm_pass(
        self,
        session_id: str,
        room_id: str,
        messages: list[LLMMessage],
        trace,
    ) -> tuple[list[LLMToolCall], str]:
        specs = self.mcp.specs()
        trace.event(
            "llm_request",
            tools=[s.qualified for s in specs],
            messages=[m.model_dump() for m in messages],
        )
        pending: list[LLMToolCall] = []
        text_chunks: list[str] = []
        async for event in self.llm.chat(messages, specs):
            if isinstance(event, LLMText):
                text_chunks.append(event.text)
                await self._broadcast(
                    room_id, AssistantDelta(session_id=session_id, text=event.text)
                )
                trace.event("assistant_delta", text=event.text)
            elif isinstance(event, LLMToolCall):
                pending.append(event)
        return pending, "".join(text_chunks)

    async def _handle_loop_exhausted(
        self, session_id: str, room_id: str, trace
    ) -> None:
        msg = "I got stuck in a tool loop and stopped. Try rephrasing."
        await self._broadcast(
            room_id, AssistantDelta(session_id=session_id, text=msg)
        )
        trace.event("tool_loop_exhausted", limit=_MAX_TOOL_LOOP)

    async def _run_tool_calls(
        self,
        session_id: str,
        room_id: str,
        envelope: CallEnvelope,
        calls: list[LLMToolCall],
        messages: list[LLMMessage],
        trace,
    ) -> None:
        for tc in calls:
            await self._broadcast(
                room_id,
                ToolCall(
                    session_id=session_id,
                    call_id=tc.call_id,
                    server=tc.server,
                    name=tc.name,
                    args=tc.args,
                ),
            )
            trace.event(
                "tool_call",
                call_id=tc.call_id,
                server=tc.server,
                name=tc.name,
                args=tc.args,
            )
            result = await self.mcp.dispatch(tc.server, tc.name, tc.args, envelope)
            await self._broadcast(
                room_id,
                ToolResult(
                    session_id=session_id,
                    call_id=tc.call_id,
                    ok=result.ok,
                    content=result.content,
                    error=result.error,
                ),
            )
            trace.event(
                "tool_result",
                call_id=tc.call_id,
                ok=result.ok,
                content=result.content,
                error=result.error,
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    tool_call_id=tc.call_id,
                    content=json.dumps(result.content) if result.ok else (result.error or "error"),
                )
            )

    async def _broadcast(self, room_id: str, msg: BaseModel) -> None:
        for cid in self.clients_in_room(room_id):
            await self.send(cid, msg)
