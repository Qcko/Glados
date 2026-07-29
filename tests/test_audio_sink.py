from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from glados.core.audio_sink import AudioSink, FrameTooShort
from glados.core.protocols import AUDIO_HEADER_LEN, AUDIO_SAMPLE_RATE


def _frame(seq: int, samples: list[int]) -> bytes:
    pcm = b"".join(struct.pack("<h", s) for s in samples)
    return struct.pack(">I", seq) + pcm


def test_first_write_creates_wav(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    assert sink.path is None

    sink.write(_frame(0, [100, -200, 300]))
    assert sink.path is not None
    assert sink.path.parent == tmp_path / "audio" / "desk-ui"
    assert sink.path.suffix == ".wav"
    assert sink.frames_written == 1
    assert sink.samples_written == 3


def test_wav_has_correct_format(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    samples = [0, 1000, -1000, 32767, -32768]
    sink.write(_frame(0, samples))
    sink.write(_frame(1, samples))
    sink.close()

    assert sink.path is not None
    with wave.open(str(sink.path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == AUDIO_SAMPLE_RATE
        assert wav.getnframes() == len(samples) * 2
        body = wav.readframes(wav.getnframes())
    expected = b"".join(struct.pack("<h", s) for s in samples) * 2
    assert body == expected


def test_drop_detection(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    sink.write(_frame(0, [0]))
    sink.write(_frame(1, [0]))
    sink.write(_frame(5, [0]))   # gap of 3
    sink.write(_frame(6, [0]))
    assert sink.dropped == 3


def test_short_frame_rejected(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    with pytest.raises(FrameTooShort):
        sink.write(b"\x00\x00\x00")


def test_odd_pcm_length_rejected(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    bad = struct.pack(">I", 0) + b"\x01\x02\x03"  # 3 bytes of "PCM" -- not int16
    with pytest.raises(FrameTooShort):
        sink.write(bad)


def test_close_is_idempotent(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    sink.write(_frame(0, [0]))
    sink.close()
    sink.close()  # should not raise


def test_close_without_write_is_noop(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    sink.close()
    assert sink.path is None
    assert not (tmp_path / "audio").exists()


def test_header_len_constant() -> None:
    assert AUDIO_HEADER_LEN == 4


@pytest.mark.parametrize("bad", ["../evil", "..\\evil", "a/b", "a\\b", "", ".", ".."])
def test_unsafe_client_id_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        AudioSink(tmp_path, bad)


def test_seq_restart_not_counted_as_drop(tmp_path: Path) -> None:
    sink = AudioSink(tmp_path, "desk-ui")
    for seq in (0, 1, 2, 3):
        sink.write(_frame(seq, [0]))
    # mic toggled off then on -- client restarts seq at 0
    for seq in (0, 1, 2):
        sink.write(_frame(seq, [0]))
    assert sink.dropped == 0
