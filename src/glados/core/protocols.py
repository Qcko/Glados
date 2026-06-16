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


class PlaybackDone(BaseModel):
    """A speaker client reports that the TTS audio for `session_id` has finished
    playing out of its buffer. Lets the server shorten the feedback gate from
    the duration estimate to the short tail cooldown (see Organizer). Only
    honored from a speaker-role client (server enforces); advisory — a missing
    one just falls back to the estimate."""

    type: Literal["playback_done"] = "playback_done"
    session_id: str


ClientMessage = Annotated[
    Hello | UserText | Interrupt | ToolConfirmResponse | PlaybackDone,
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
    actually finish?").

    `confabulated` is the special case where the model claimed an *action* was
    done while dispatching no tools at all — a fabricated completion, the
    signature of a poisoned history. The organizer suppresses the false claim
    from the spoken reply and never commits it to the hot buffer."""

    type: Literal["turn_outcome"] = "turn_outcome"
    session_id: str
    outcome: Literal["done", "needs-user", "failed", "confabulated"]


class RouteNotice(BaseModel):
    """Which brain (primary or specialist) handled the turn, broadcast when the
    v2.6 local multi-model router is active. `escalated=True` marks the *second*
    notice of a turn: the primary path produced a `failed` outcome and the
    organizer is retrying on the specialist. The UI surfaces which model ran so
    the operator can see when a turn fell through to the heavier path."""

    type: Literal["route_notice"] = "route_notice"
    session_id: str
    target: Literal["primary", "specialist"]
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


class MemoryBlockNotice(BaseModel):
    """Operator-facing notice that a *trusted* MCP server shipped lessons that
    did not clear the LocalGuard hash-approval gate, so **nothing was injected**
    (ARCH §14, BLOCK-notice surface). Carries metadata only — `source` id,
    `sha256` of the blob, character `length`, and LocalGuard's `reason`. It
    holds **none of the untrusted blob bytes** and is never routed through the
    assistant LLM or TTS, so it is safe on any surface; the UI renders it as an
    admin banner with an affordance to start a review. It is emitted at load
    time (before any client connects), so it is also pushed to `ui`-role clients
    on connect and exposed at `GET /admin/memory`. Granting a review/approve is
    a separate, high-friction desktop action — never this notice, never voice."""

    type: Literal["memory_block_notice"] = "memory_block_notice"
    source: str
    sha256: str
    length: int
    reason: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


# ---- Admin observe channel (loopback-only admin port) -------------------
# A separate, loopback-bound surface (core/server.py admin_app, default
# 127.0.0.1:9765) lets an operator watch any room's conversation as text for
# debugging. It NEVER rides the LAN-facing /ws/v1, and the house-wide observe
# capability answers only on loopback + behind a distinct admin secret
# (ARCHITECTURE §9).


class AdminHello(BaseModel):
    """First message on the admin channel: authenticate with the admin
    secret (constant-time compared server-side). Distinct from the room
    `Hello` — the admin surface has no room/role binding."""

    type: Literal["admin_hello"] = "admin_hello"
    token: str


class ObserveRoom(BaseModel):
    """Admin asks to observe `room_id` (read-only). `room_id=None` stops
    observing — closes the tab server-side so a dead subscription can't
    keep fanning out."""

    type: Literal["observe_room"] = "observe_room"
    room_id: str | None = None


class HelloAck(BaseModel):
    """Server reply to a verified AdminHello: the rooms available to observe
    (derived from rooms.toml bindings). The admin client builds its room
    picker from this — no separate HTTP route, so nothing admin leaks onto
    the LAN-facing app."""

    type: Literal["hello_ack"] = "hello_ack"
    rooms: list[str]


class ObservedEvent(BaseModel):
    """A room's forwarded conversation event, wrapped so the admin client can
    attribute it to a room (the inner events carry only `session_id`). `event`
    is a serialized server message; only an allowlist of text turn-events is
    ever forwarded (audio, tool-confirm, and memory notices are not — see
    `server.py` `_make_notify_observers`)."""

    type: Literal["observed_event"] = "observed_event"
    room_id: str
    event: dict


AdminClientMessage = Annotated[
    AdminHello | ObserveRoom,
    Field(discriminator="type"),
]


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
    | MemoryBlockNotice
    | ErrorMessage,
    Field(discriminator="type"),
]
