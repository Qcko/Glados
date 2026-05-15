"""Wire protocol for /ws/v1.

Every message is a Pydantic model with a literal `type` field, so they form a
discriminated union. Adapter Protocols (STT, TTS, LLM, WakeWord) live in
adapters.py — keeping them out of the wire schema.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


Role = Literal["mic", "speaker", "ui"]


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    client_id: str
    room_id: str
    role: Role
    token: str


class UserText(BaseModel):
    type: Literal["user_text"] = "user_text"
    text: str


class Interrupt(BaseModel):
    type: Literal["interrupt"] = "interrupt"
    session_id: str


ClientMessage = Annotated[
    Hello | UserText | Interrupt,
    Field(discriminator="type"),
]


# Audio is sent as raw binary WebSocket frames, not JSON. Frame layout:
#   bytes 0..4   : big-endian uint32 sequence number
#   bytes 4..end : PCM16-LE samples at AUDIO_SAMPLE_RATE Hz, mono
# Sample rate is a constant on both ends — keeping it implicit avoids
# per-frame header bloat and lets the server treat every byte beyond
# the prefix as audio samples.
AUDIO_SAMPLE_RATE = 16_000
AUDIO_HEADER_LEN = 4


class Welcome(BaseModel):
    type: Literal["welcome"] = "welcome"
    session_id: str


class AssistantDelta(BaseModel):
    type: Literal["assistant_delta"] = "assistant_delta"
    session_id: str
    text: str


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    session_id: str
    call_id: str
    server: str
    name: str
    args: dict


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    session_id: str
    call_id: str
    ok: bool
    content: dict | None = None
    error: str | None = None


class TtsChunk(BaseModel):
    type: Literal["tts_chunk"] = "tts_chunk"
    session_id: str
    seq: int
    pcm_b64: str


class Done(BaseModel):
    type: Literal["done"] = "done"
    session_id: str


class Cancelled(BaseModel):
    type: Literal["cancelled"] = "cancelled"
    session_id: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


ServerMessage = Annotated[
    Welcome | AssistantDelta | ToolCall | ToolResult | TtsChunk | Done | Cancelled | ErrorMessage,
    Field(discriminator="type"),
]
