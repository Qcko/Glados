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


class ToolConfirmResponse(BaseModel):
    """Reply to a ToolConfirmRequest. `granted=False` denies the tool
    call; the LLM sees `MCPCallResult(ok=False, error="user denied")`
    and can recover. The Organizer enforces that the responder is in the
    originating room — replies from other rooms are dropped."""

    type: Literal["tool_confirm_response"] = "tool_confirm_response"
    request_id: str
    granted: bool


ClientMessage = Annotated[
    Hello | UserText | Interrupt | ToolConfirmResponse,
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


class UserTranscript(BaseModel):
    """What the server believes the user said for this turn. Broadcast
    once per turn, right after Welcome. `source` distinguishes a typed
    `user_text` ingress from an audio-derived STT transcript — the UI
    shows the latter differently so STT mistranscriptions are visible
    at a glance (the whole point: catching e.g. Czech misdetected as
    French without having to grep traces)."""

    type: Literal["user_transcript"] = "user_transcript"
    session_id: str
    text: str
    source: Literal["voice", "text"]


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
    sample_rate: int
    pcm_b64: str


class Done(BaseModel):
    type: Literal["done"] = "done"
    session_id: str


class Cancelled(BaseModel):
    type: Literal["cancelled"] = "cancelled"
    session_id: str


class TurnOutcome(BaseModel):
    """Typed verdict on how the turn ended, broadcast right before `Done`.

    Derived deterministically from observed tool results (see
    `core/turn_outcome.py`), never from the model's self-report. The UI can
    surface a failed/needs-user turn that the model narrated as success, and
    the v2.6 router consumes it as an escalation input ("did the local model
    actually finish?")."""

    type: Literal["turn_outcome"] = "turn_outcome"
    session_id: str
    outcome: Literal["done", "needs-user", "failed"]


class RouteNotice(BaseModel):
    """Which brain (local or cloud) handled the turn, broadcast when the v2.6
    hybrid router is active. `escalated=True` marks the *second* notice of a
    turn: the local path produced a `failed` outcome and the organizer is
    retrying on cloud. The UI surfaces the cloud path explicitly because, per
    ARCHITECTURE §9, the cloud path sends tool arguments/results externally —
    the user should see when their data crossed that boundary."""

    type: Literal["route_notice"] = "route_notice"
    session_id: str
    target: Literal["local", "cloud"]
    reason: str
    escalated: bool = False


class ToolConfirmRequest(BaseModel):
    """Sent to clients in the originating room when the LLM tries to
    call a tool whose ToolSpec.requires_confirmation is True. Any client
    in the room can answer; the first response wins. The UI renders a
    confirm modal showing `tool` and `args_summary`. `ttl_s` is the
    server's deny-deadline; the UI may render a countdown but the
    server enforces the truth."""

    type: Literal["tool_confirm_request"] = "tool_confirm_request"
    session_id: str
    request_id: str
    tool: str
    args_summary: dict
    ttl_s: float


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


ServerMessage = Annotated[
    Welcome
    | UserTranscript
    | AssistantDelta
    | ToolCall
    | ToolResult
    | TtsChunk
    | Done
    | Cancelled
    | TurnOutcome
    | RouteNotice
    | ToolConfirmRequest
    | ErrorMessage,
    Field(discriminator="type"),
]
