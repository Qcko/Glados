"""TurnLatency: the per-turn breakdown of speech-end-to-first-audio."""

from __future__ import annotations

import logging

from glados.core.turn_latency import SpeechTiming, TurnLatency


class _Trace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, name: str, **fields) -> None:
        self.events.append((name, fields))


def test_breakdown_adds_up_from_speech_end(caplog) -> None:
    speech = SpeechTiming(segment_end_at=10.0, hangover_ms=800, stt_ms=2000)
    latency = TurnLatency("kitchen", speech, started_at=12.1)
    latency.add_tool(3.5)
    trace = _Trace()
    with caplog.at_level(logging.INFO):
        latency.report(synth_started_at=17.1, first_chunk_at=17.4, trace=trace)
    name, numbers = trace.events[0]
    assert name == "turn_latency"
    assert numbers == {
        "hangover_ms": 800, "stt_ms": 2000, "queue_ms": 100, "think_ms": 5000,
        "tool_ms": 3500, "synth_ms": 300, "total_ms": 8200,
    }
    assert "speech_end_to_audio=8200ms" in caplog.text


def test_reports_once_per_turn() -> None:
    latency = TurnLatency("desk", SpeechTiming(1.0, 0, 0), started_at=1.0)
    trace = _Trace()
    latency.report(1.0, 1.1, trace)
    latency.report(2.0, 2.1, trace)
    assert len(trace.events) == 1
