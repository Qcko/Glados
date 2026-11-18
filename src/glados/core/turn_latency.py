"""Per-turn latency: where the wait between "user stopped talking" and
"GLaDOS starts talking" goes.

The audio pipeline stamps a `SpeechTiming` when the VAD closes a segment and
STT returns; the Organizer opens a `TurnLatency` when the turn starts, adds
tool time as calls return, and reports once, on the first TTS chunk. One INFO
line per voice turn, plus the same numbers as a trace event.

`stt` runs from segment end to text, so it includes waiting for the wake
decision and for the Whisper lock behind an earlier segment; `queue` includes
waiting for the previous segment's dispatch and the room's turn queue.

All stamps are event-loop time (`loop.time()`), the clock the pipeline and the
Organizer already share.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechTiming:
    """Measured in the audio pipeline for one segment."""

    segment_end_at: float
    hangover_ms: int
    stt_ms: int


@dataclass
class TurnLatency:
    room_id: str
    speech: SpeechTiming
    started_at: float
    tool_ms: int = 0
    reported: bool = field(default=False)

    def add_tool(self, elapsed_s: float) -> None:
        self.tool_ms += round(elapsed_s * 1000)

    def report(self, synth_started_at: float, first_chunk_at: float, trace) -> None:
        if self.reported:
            return
        self.reported = True
        numbers = self._numbers(synth_started_at, first_chunk_at)
        log.info(
            "turn latency room=%s hangover=%dms stt=%dms queue=%dms think=%dms "
            "(tool calls summed %dms) synth=%dms -> speech_end_to_audio=%dms",
            self.room_id, numbers["hangover_ms"], numbers["stt_ms"],
            numbers["queue_ms"], numbers["think_ms"], numbers["tool_ms"],
            numbers["synth_ms"], numbers["total_ms"],
        )
        trace.event("turn_latency", **numbers)

    def _numbers(self, synth_started_at: float, first_chunk_at: float) -> dict[str, int]:
        stt_done_at = self.speech.segment_end_at + self.speech.stt_ms / 1000
        speech_end_at = self.speech.segment_end_at - self.speech.hangover_ms / 1000
        return {
            "hangover_ms": self.speech.hangover_ms,
            "stt_ms": self.speech.stt_ms,
            "queue_ms": _ms(self.started_at - stt_done_at),
            "think_ms": _ms(synth_started_at - self.started_at),
            "tool_ms": self.tool_ms,
            "synth_ms": _ms(first_chunk_at - synth_started_at),
            "total_ms": _ms(first_chunk_at - speech_end_at),
        }


def _ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


# The latency record of the voice turn running in THIS task, like the
# Organizer's per-turn cart: a context variable so a superseded turn's late
# tool return cannot count into its successor.
TURN_LATENCY: ContextVar[TurnLatency | None] = ContextVar("turn_latency", default=None)
