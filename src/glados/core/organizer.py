"""Organizer: the only place sessions, queueing, and egress routing live.

v0 step 2 scope: ingress tagging via `ClientBinding`, single-turn sessions
(no continuation yet), per-session tool-calling loop, egress fan-out by
`room_id`. Dedup, fingerprinting, and per-session FIFO queueing land when
audio arrives in v1.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import aclosing
from typing import Awaitable, Callable

from pydantic import BaseModel

from ..mcp.registry import CallEnvelope, MCPRegistry
from .adapters import LLM, TTS, LLMMessage, LLMText, LLMToolCall
from .config import ClientBinding
from .protocols import (
    AssistantDelta,
    Cancelled,
    Done,
    ToolCall,
    ToolResult,
    TtsChunk,
    Welcome,
)
from .sessions import SessionRegistry
from .traces import TraceStore

log = logging.getLogger(__name__)

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
        tts: TTS | None = None,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.mcp = mcp
        self.traces = traces
        self.sessions = sessions
        self.send = send
        self.binding_for_client = binding_for_client
        self.clients_in_room = clients_in_room
        # session_id -> (task, room_id). Lets handle_interrupt cancel the
        # right turn and route the Cancelled broadcast to the right room.
        self._inflight: dict[str, tuple[asyncio.Task, str]] = {}

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
        task = asyncio.current_task()
        if task is not None:
            self._inflight[session.session_id] = (task, session.room_id)
        trace = self.traces.open(session.session_id)
        cancelled = False
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
            final_text = ""
            for _ in range(_MAX_TOOL_LOOP):
                pending_calls, assistant_text = await self._run_one_llm_pass(
                    session.session_id, session.room_id, messages, trace
                )
                if not pending_calls:
                    final_text = assistant_text
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
                final_text = await self._handle_loop_exhausted(
                    session.session_id, session.room_id, trace
                )
            await self._speak(session.session_id, session.room_id, final_text, trace)
            await self._broadcast(session.room_id, Done(session_id=session.session_id))
            trace.event("done")
        except asyncio.CancelledError:
            cancelled = True
            trace.event("cancelled")
        finally:
            self._inflight.pop(session.session_id, None)
            # Close the trace before any further await so a re-cancel during
            # shutdown can't strand the file handle. Broadcast is shielded
            # for the same reason — without it a second cancel would
            # suppress Cancelled and leave the room hanging.
            trace.close()
            if cancelled:
                await asyncio.shield(
                    self._broadcast(
                        session.room_id,
                        Cancelled(session_id=session.session_id),
                    )
                )

    async def handle_interrupt(self, client_id: str, session_id: str) -> None:
        binding = self.binding_for_client(client_id)
        if binding is None:
            return
        entry = self._inflight.get(session_id)
        if entry is None:
            return  # turn already finished or never existed — no-op
        task, room_id = entry
        if room_id != binding.room_id:
            log.warning(
                "interrupt rejected: client %s in room %s tried to cancel session %s in room %s",
                client_id, binding.room_id, session_id, room_id,
            )
            return
        task.cancel()

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
        # aclosing guarantees the upstream HTTP stream gets aclose()'d on
        # CancelledError — otherwise Ollama keeps generating tokens we'll
        # never read (ARCH §6: cancellation must propagate end-to-end).
        async with aclosing(self.llm.chat(messages, specs)) as stream:
            async for event in stream:
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
    ) -> str:
        msg = "I got stuck in a tool loop and stopped. Try rephrasing."
        await self._broadcast(
            room_id, AssistantDelta(session_id=session_id, text=msg)
        )
        trace.event("tool_loop_exhausted", limit=_MAX_TOOL_LOOP)
        return msg

    async def _speak(
        self, session_id: str, room_id: str, text: str, trace
    ) -> None:
        if self.tts is None or not text.strip():
            return
        seq = 0
        try:
            async with aclosing(self.tts.synthesize(text)) as stream:
                async for chunk in stream:
                    await self._broadcast(
                        room_id,
                        TtsChunk(
                            session_id=session_id,
                            seq=seq,
                            sample_rate=chunk.sample_rate,
                            pcm_b64=base64.b64encode(chunk.pcm).decode("ascii"),
                        ),
                    )
                    trace.event(
                        "tts_chunk",
                        seq=seq,
                        samples=len(chunk.pcm) // 2,
                        sample_rate=chunk.sample_rate,
                    )
                    seq += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            # TTS is a side-channel — don't break the turn if synth blows up.
            log.exception("tts synthesize failed")
            trace.event("tts_error")

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
