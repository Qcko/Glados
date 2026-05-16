"""Internal adapter Protocols + LLM event types.

Wire-protocol message models live in `protocols.py`; this module is for the
shapes the orchestrator passes between components.
"""

from __future__ import annotations

from typing import Annotated, AsyncIterator, Literal, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    server: str
    name: str
    description: str
    parameters: dict

    @property
    def qualified(self) -> str:
        return f"{self.server}.{self.name}"


class LLMText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class LLMToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    server: str
    name: str
    args: dict


LLMEvent = Annotated[LLMText | LLMToolCall, Field(discriminator="type")]


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None


class LLM(Protocol):
    def chat(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        """Stream `LLMEvent`s for one turn. May be an async generator or any
        coroutine that returns an async iterator. Caller iterates with
        `async for`."""
        ...


class VadStart(BaseModel):
    type: Literal["start"] = "start"


class VadEnd(BaseModel):
    type: Literal["end"] = "end"
    pcm: bytes


VadEvent = Annotated[VadStart | VadEnd, Field(discriminator="type")]


class VAD(Protocol):
    """Voice-activity detector over a stream of 16 kHz int16 PCM bytes.

    `feed` consumes a chunk of PCM and returns zero or more boundary events
    in arrival order. The detector buffers samples internally and emits a
    `VadEnd` carrying the full utterance PCM when speech ends. `reset`
    clears any in-flight utterance (called on disconnect)."""

    def feed(self, pcm: bytes) -> list["VadEvent"]: ...
    def reset(self) -> None: ...


class STT(Protocol):
    """Transcribe one utterance of 16 kHz mono int16 PCM into text.

    The v1 step 2 contract is one-shot (full utterance in, text out). The
    streaming-partials variant from ARCHITECTURE §6 is deferred until a
    backend wants it; wrapping is trivial."""

    async def transcribe(self, pcm: bytes) -> str: ...
