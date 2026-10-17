"""Wake word end to end: pipeline hook, Organizer admission, detector backend,
boot-time model checks (DESIGN.md, wake word)."""

from __future__ import annotations

import asyncio
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from glados.audio.pipeline import AudioPipeline
from glados.audio.stt.fake import FakeSTT
from glados.audio.vad.fake import FakeVAD
from glados.audio.wake.fake import FakeWakeWord
from glados.audio.wake.openwakeword import OpenWakeWord, WakeModelError, verify_model_files
from glados.brain.llm.fake import FakeLLM
from glados.core.config import ClientBinding, GladosConfig, RoomPolicy, RoomsConfig, WakeConfig
from glados.core.server import _build_wake, _wake_hook
from glados.core.wake import Admission, WakeGate, WakeTimings

from .organizer_harness import desk_organizer

KITCHEN = ClientBinding(client_id="kitchen-mic", room_id="kitchen", role="mic")
KITCHEN_2 = ClientBinding(client_id="kitchen-mic-2", room_id="kitchen", role="mic")
LIVING = ClientBinding(client_id="living-mic", room_id="living", role="mic")
KITCHEN_SPEAKER = ClientBinding(client_id="kitchen-spk", room_id="kitchen", role="speaker")
BINDINGS = (KITCHEN, KITCHEN_2, LIVING, KITCHEN_SPEAKER)


def _frame(seq: int, samples: list[int]) -> bytes:
    return struct.pack(">I", seq) + b"".join(struct.pack("<h", s) for s in samples)


def _gate(dedup_s: float = 0.05, armed_s: float = 8.0) -> WakeGate:
    return WakeGate(
        timings=WakeTimings(armed_window_s=armed_s, followup_window_s=6.0, dedup_window_s=dedup_s),
        mode_for_room=lambda _room: "required",
    )


def _transcripts(h) -> list[str]:
    return [m["text"] for cid, m in h.sink
            if m.get("type") == "user_transcript" and cid in ("kitchen-spk", "living-mic")]


async def _settle(h) -> None:
    await asyncio.sleep(0.1)
    await h.org.flush()


# ---- AudioPipeline wake hook -------------------------------------------


class _RecordingHook:
    def __init__(self, detector, verdict: Admission | None = Admission("armed")) -> None:
        self.detector = detector
        self.cooldown_s = 1.5
        self.hits: list[tuple[float, float, float]] = []
        self.faults = 0
        self.decided: list[tuple[float, float]] = []
        self._verdict = verdict

    def on_hit(self, score, energy, at) -> None:
        self.hits.append((score, energy, at))

    def on_fault(self) -> None:
        self.faults += 1

    async def decide(self, start, end):
        self.decided.append((start, end))
        return self._verdict


def _pipeline(hook, stt=None, seen=None) -> AudioPipeline:
    seen = seen if seen is not None else []

    async def on_utterance(text, captured_at, admission=None):
        seen.append((text, admission))

    return AudioPipeline(
        sink=None, vad=FakeVAD(utterance_samples=4), stt=stt or FakeSTT("hey glados lights"),
        on_utterance=on_utterance, wake=hook,
    )


async def test_pipeline_reports_hit_with_energy_and_arrival_time() -> None:
    hook = _RecordingHook(FakeWakeWord({0: 0.9}))
    pipe = _pipeline(hook)
    await pipe.feed_frame(_frame(0, [16384, -16384]))
    score, energy, _at = hook.hits[0]
    assert score == 0.9
    assert energy == pytest.approx(-6.02, abs=0.1)
    await pipe.close()


async def test_pipeline_cooldown_collapses_a_burst_of_hits() -> None:
    hook = _RecordingHook(FakeWakeWord({0: 0.7, 1: 0.8, 2: 0.9}))
    pipe = _pipeline(hook)
    for i in range(3):
        await pipe.feed_frame(_frame(i, [1]))
    assert len(hook.hits) == 1
    await pipe.close()


async def test_refused_segment_is_never_transcribed() -> None:
    transcribed: list[bytes] = []

    def stt_text(pcm: bytes) -> str:
        transcribed.append(pcm)
        return "private chat"

    seen: list = []
    hook = _RecordingHook(FakeWakeWord(), verdict=None)
    pipe = _pipeline(hook, stt=FakeSTT(stt_text), seen=seen)
    await pipe.feed_frame(_frame(0, [1, 2, 3, 4]))
    await pipe.drain()
    assert hook.decided and transcribed == [] and seen == []
    await pipe.close()


async def test_admitted_segment_carries_its_admission() -> None:
    seen: list = []
    pipe = _pipeline(_RecordingHook(FakeWakeWord()), seen=seen)
    await pipe.feed_frame(_frame(0, [1, 2, 3, 4]))
    await pipe.drain()
    assert seen == [("hey glados lights", Admission("armed"))]
    await pipe.close()


async def test_detector_fault_is_reported_once_and_detection_stops() -> None:
    detector = FakeWakeWord({1: 0.9}, fail_at=0)
    hook = _RecordingHook(detector)
    pipe = _pipeline(hook)
    await pipe.feed_frame(_frame(0, [1]))
    await pipe.feed_frame(_frame(1, [1]))
    assert hook.faults == 1 and hook.hits == []
    await pipe.close()


async def test_close_resets_the_detector() -> None:
    detector = FakeWakeWord()
    pipe = _pipeline(_RecordingHook(detector))
    await pipe.close()
    assert detector.resets == 1


# ---- Organizer admission -----------------------------------------------


async def _say(org, client: str, text: str, start: float, end: float) -> None:
    admission = await org.wake_decide(client, start, end)
    if admission is not None:
        await org.handle_audio_text(client, text, start, admission)


async def test_unwoken_speech_never_becomes_a_turn(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        await _say(h.org, "kitchen-mic", "what time is it", now, now + 1)
        await _settle(h)
        assert _transcripts(h) == []


async def test_one_breath_wake_runs_the_stripped_command(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now + 0.5)
        await _say(h.org, "kitchen-mic", "Hey GLaDOS, what time is it", now, now + 1)
        await _settle(h)
        assert _transcripts(h) == ["what time is it"]


async def test_wake_only_segment_arms_without_a_turn_then_command_runs(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now + 0.5)
        await _say(h.org, "kitchen-mic", "Hey GLaDOS.", now, now + 0.6)
        await _settle(h)
        assert _transcripts(h) == []
        later = asyncio.get_running_loop().time()
        await _say(h.org, "kitchen-mic", "lights on", later, later + 1)
        await _settle(h)
        assert _transcripts(h) == ["lights on"]


async def test_two_rooms_hear_one_wake_and_only_the_loudest_runs_it(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("living-mic", 0.9, -40.0, now + 0.5)
        h.org.wake_hit("kitchen-mic", 0.9, -18.0, now + 0.52)
        await asyncio.gather(
            _say(h.org, "kitchen-mic", "hey glados lights", now, now + 1),
            _say(h.org, "living-mic", "hey glados lights", now, now + 1),
        )
        await _settle(h)
        assert _transcripts(h) == ["lights"]
        rooms = {cid for cid, m in h.sink if m.get("type") == "user_transcript"}
        assert "living-mic" not in rooms


async def test_short_command_waits_for_its_own_arbitration_window(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(0.3),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now)
        await _say(h.org, "kitchen-mic", "hey glados lights", now - 0.5, now)
        await _settle(h)
        assert _transcripts(h) == ["lights"]


async def test_hit_during_any_rooms_playback_is_ignored(tmp_path: Path) -> None:
    from glados.core.organizer import _TtsGate

    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org._tts_gate["living"] = _TtsGate(phase="sending", session_id="s", closed_until=now + 30.0)
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now + 0.5)
        await _say(h.org, "kitchen-mic", "hey glados lights", now, now + 1)
        await _settle(h)
        assert _transcripts(h) == []


async def test_speaker_role_audio_cannot_reach_a_turn(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS, wake_gate=_gate(),
                              wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now + 0.5)
        await _settle(h)
        await _say(h.org, "kitchen-spk", "buy everything", now + 0.6, now + 1)
        await _settle(h)
        assert _transcripts(h) == []


async def test_follow_up_after_a_reply_needs_no_wake(tmp_path: Path) -> None:
    async with desk_organizer(tmp_path, llm=FakeLLM(), bindings=BINDINGS,
                              wake_gate=_gate(armed_s=1.0), wake_aliases=["hey glados"]) as h:
        now = asyncio.get_running_loop().time()
        h.org.wake_hit("kitchen-mic", 0.9, -20.0, now + 0.5)
        await _say(h.org, "kitchen-mic", "hey glados what time is it", now, now + 1)
        await _settle(h)
        later = asyncio.get_running_loop().time() + 3.0
        await _say(h.org, "kitchen-mic", "and the date", later, later + 1)
        await _settle(h)
        assert _transcripts(h) == ["what time is it", "and the date"]


# ---- openWakeWord backend ----------------------------------------------


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.frames: list[int] = []
        self.resets = 0

    def predict(self, frame):
        self.frames.append(len(frame))
        return {"hey_jarvis": self._scores.pop(0) if self._scores else 0.0}

    def reset(self) -> None:
        self.resets += 1


def _oww(model: _FakeModel, threshold: float = 0.5) -> OpenWakeWord:
    return OpenWakeWord(
        model_path=Path("m"), melspec_path=Path("a"), embedding_path=Path("b"),
        threshold=threshold, model_factory=lambda *_: model,
    )


def test_openwakeword_feeds_whole_80ms_frames_and_thresholds() -> None:
    model = _FakeModel([0.2, 0.9])
    oww = _oww(model)
    assert oww.feed(np.zeros(1000, dtype="<i2").tobytes()) == []
    assert oww.feed(np.zeros(1560, dtype="<i2").tobytes()) == [0.9]
    assert model.frames == [1280, 1280]


def test_openwakeword_reset_clears_buffer_and_model() -> None:
    model = _FakeModel([])
    oww = _oww(model)
    oww.feed(np.zeros(100, dtype="<i2").tobytes())
    oww.reset()
    oww.feed(np.zeros(1200, dtype="<i2").tobytes())
    assert model.frames == [] and model.resets == 1


def test_verify_model_files_accepts_matching_pins(tmp_path: Path) -> None:
    (tmp_path / "w.onnx").write_bytes(b"model")
    verify_model_files(tmp_path, {"w.onnx": hashlib.sha256(b"model").hexdigest()})


@pytest.mark.parametrize("pins", [{}, {"w.onnx": "0" * 64}, {"missing.onnx": "0" * 64}])
def test_verify_model_files_refuses(tmp_path: Path, pins: dict[str, str]) -> None:
    (tmp_path / "w.onnx").write_bytes(b"model")
    with pytest.raises(WakeModelError):
        verify_model_files(tmp_path, pins)


# ---- boot wiring -------------------------------------------------------


def _rooms(**policy) -> RoomsConfig:
    rooms = [RoomPolicy(room_id="kitchen", **policy)] if policy else []
    return RoomsConfig(clients=[KITCHEN, LIVING, KITCHEN_SPEAKER], rooms=rooms)


def test_no_wiring_when_no_room_requires_a_wake(tmp_path: Path) -> None:
    assert _build_wake(GladosConfig(), _rooms(), tmp_path) is None


def test_per_room_override_requires_wake_in_that_room_only(tmp_path: Path) -> None:
    cfg = GladosConfig(wake=WakeConfig(backend="fake"))
    wiring = _build_wake(cfg, _rooms(wake_mode="required"), tmp_path)
    assert wiring.gate.mode("kitchen") == "required"
    assert wiring.gate.mode("living") == "off"


def test_required_room_with_unpinned_model_refuses_boot(tmp_path: Path) -> None:
    cfg = GladosConfig(wake=WakeConfig(mode="required"))
    with pytest.raises(RuntimeError):
        _build_wake(cfg, _rooms(), tmp_path)


def test_hook_only_for_required_rooms_and_never_wakes_for_non_mic(tmp_path: Path) -> None:
    cfg = GladosConfig(wake=WakeConfig(backend="fake"))
    wiring = _build_wake(cfg, _rooms(wake_mode="required"), tmp_path)
    assert _wake_hook(wiring, organizer=None, binding=LIVING) is None
    assert isinstance(_wake_hook(wiring, None, KITCHEN).detector, FakeWakeWord)
    assert _wake_hook(wiring, None, KITCHEN_SPEAKER).detector.feed(b"\x00\x00") == []
