"""Internal adapter Protocols + LLM event types.

Wire-protocol message models live in `protocols.py`; this module is for the
shapes the orchestrator passes between components.
"""

from __future__ import annotations

from typing import Annotated, AsyncIterator, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    # Reject unknown keys on construction. Catches typos in stdio MCP
    # servers' `tools/list` payloads — a misspelled `require_confirmation`
    # would otherwise silently default to False and gate nothing.
    model_config = ConfigDict(extra="forbid")

    server: str
    name: str
    description: str
    parameters: dict
    # When True, the tool returns content from outside the local trust
    # boundary (web fetches, scraped pages, third-party APIs). The
    # Organizer wraps the result in <external>...</external> delimiters
    # before feeding it back to the LLM. Pair with a system-prompt rule
    # that instructions inside <external> are data, not commands.
    # See ARCHITECTURE.md §7 untrusted-content discipline.
    untrusted: bool = False
    # When True, the Organizer broadcasts a ToolConfirmRequest to the
    # originating room before dispatch and waits for ToolConfirmResponse
    # (granted=True) within the deny timeout. Hard-coded per tool by the
    # author — not LLM-decided. Per ARCH §7 permission gates: any
    # side-effecting tool (cart writes, checkout, login, money) MUST be
    # gated. Default False so today's read-only tools (echo, time, etc.)
    # are unchanged.
    requires_confirmation: bool = False
    # True when the tool mutates external state (cart writes, checkout, login).
    # Distinct from requires_confirmation: a side-effecting tool can be
    # intentionally un-gated (Dunnes cart writes are), so confirmation is NOT a
    # reliable "did something change" signal. The turn-outcome goal-check reads
    # this to tell a real action from a read/search. Until the per-tool risk
    # manifest (class=write/payment) is wired, it's set via the servers.toml
    # overlay; requires_confirmation=True implies mutating too (see Organizer).
    mutating: bool = False
    # Per-tool override for the registry's dispatch timeout. None falls back
    # to the registry default (8s). Selenium-driven scrapers (Dunnes, etc.)
    # need ~30s for a page load — bake the override into the tool's spec
    # rather than threading a timeout argument through every call site.
    timeout_s: float | None = None

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


class TtsChunkOut(BaseModel):
    """One audio chunk emitted by a TTS backend.

    `pcm` is int16-LE mono samples at `sample_rate` Hz. Backends decide
    chunk granularity (Piper yields one chunk per sentence)."""

    pcm: bytes
    sample_rate: int


class TTS(Protocol):
    """Speech synthesis. Streams PCM chunks for one input text.

    The v1 step 3 contract is one-shot text -> async iterator of chunks.
    Chunk granularity is backend-defined; callers must not assume a
    fixed size. Implementations are expected to run their blocking
    inference in `asyncio.to_thread` so the event loop stays responsive."""

    def synthesize(self, text: str) -> AsyncIterator[TtsChunkOut]: ...
