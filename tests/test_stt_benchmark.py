"""Harness logic for the STT benchmark (scripts/stt_benchmark.py).

These tests exercise the benchmark *plumbing* — manifest reading, WAV
loading + format validation, latency capture, summary statistics, JSON
shape — without paying the faster-whisper download. A FakeSTT supplies
canned hypotheses so the result is deterministic.
"""

from __future__ import annotations

import json
import struct
import sys
import wave
from importlib import util
from pathlib import Path

import pytest


def _load_benchmark_module():
    """Load scripts/stt_benchmark.py without making `scripts` a package."""
    path = Path(__file__).parent.parent / "scripts" / "stt_benchmark.py"
    spec = util.spec_from_file_location("stt_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules["stt_benchmark"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def _write_wav(path: Path, samples: int = 16_000) -> None:
    """16-kHz mono 16-bit PCM silence, `samples` long (default 1 s)."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(b"".join(struct.pack("<h", 0) for _ in range(samples)))


class _ScriptedSTT:
    """Returns hypotheses in the order they were configured."""

    def __init__(self, hypotheses: list[str]) -> None:
        self._hypotheses = list(hypotheses)
        self._i = 0

    async def transcribe(self, pcm: bytes) -> str:
        out = self._hypotheses[self._i]
        self._i += 1
        return out


def test_load_pcm_validates_format(tmp_path: Path) -> None:
    good = tmp_path / "good.wav"
    _write_wav(good, samples=8_000)
    pcm, duration_s = bench._load_pcm(good)
    assert duration_s == pytest.approx(0.5)
    assert len(pcm) == 8_000 * 2


def test_load_pcm_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    with wave.open(str(bad), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44_100)
        wf.writeframes(b"\x00" * 200)
    with pytest.raises(ValueError, match="16-kHz"):
        bench._load_pcm(bad)


def test_percentile_matches_known_values() -> None:
    assert bench._percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)
    assert bench._percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert bench._percentile([], 95) == 0.0
    assert bench._percentile([7.0], 50) == 7.0


@pytest.mark.asyncio
async def test_benchmark_end_to_end(tmp_path: Path) -> None:
    """Two clips, one perfect transcription, one with a substitution.
    Verifies result shape, corpus_wer, and per-clip error counts."""
    clip_a = tmp_path / "a.wav"
    clip_b = tmp_path / "b.wav"
    _write_wav(clip_a, samples=16_000)   # 1.0 s
    _write_wav(clip_b, samples=32_000)   # 2.0 s

    manifest = [
        {"audio": str(clip_a), "text": "the quick brown fox"},
        {"audio": str(clip_b), "text": "hello world"},
    ]
    stt = _ScriptedSTT([
        "the quick brown fox",   # WER 0
        "hello earth",           # 1 substitution / 2 ref words -> WER 0.5
    ])

    result = await bench.benchmark(stt, manifest, metadata={"model": "fake"})
    assert result["model"] == "fake"
    assert result["n_clips"] == 2

    s = result["summary"]
    # corpus_wer = total_errors / total_ref_words = 1 / (4 + 2) = 1/6
    assert s["corpus_wer"] == pytest.approx(1 / 6)
    assert s["total_errors"] == 1
    assert s["total_ref_words"] == 6
    # Mean of per-clip WERs: (0 + 0.5) / 2 = 0.25
    assert s["wer_mean"] == pytest.approx(0.25)
    # corpus_rtf = sum(latency) / sum(audio); FakeSTT latency ~0,
    # so corpus_rtf is near 0 but the field must be present + numeric.
    assert "corpus_rtf" in s
    assert isinstance(s["corpus_rtf"], float)

    clip0, clip1 = result["clips"]
    assert clip0["wer"] == 0.0
    assert clip0["audio_ms"] == pytest.approx(1000.0)
    assert clip1["substitutions"] == 1
    assert clip1["audio_ms"] == pytest.approx(2000.0)
    assert clip1["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_warmup_consumes_first_n_transcriptions(tmp_path: Path) -> None:
    """`warmup=1` must consume the first scripted hypothesis silently —
    recorded clips start from the second."""
    clip = tmp_path / "a.wav"
    _write_wav(clip, samples=16_000)
    manifest = [
        {"audio": str(clip), "text": "first"},
        {"audio": str(clip), "text": "second"},
    ]
    stt = _ScriptedSTT([
        "discarded warmup output",   # consumed by warmup
        "first",                     # recorded clip 0
        "second",                    # recorded clip 1
    ])
    result = await bench.benchmark(stt, manifest, metadata={}, warmup=1)
    # Both recorded clips transcribed correctly — proves warmup ate
    # the bogus hypothesis without recording it.
    assert result["n_clips"] == 2
    assert [c["hypothesis"] for c in result["clips"]] == ["first", "second"]
    assert result["summary"]["corpus_wer"] == 0.0


def test_read_manifest_skips_blank_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "m.jsonl"
    f.write_text(
        '\n'
        '# leading comment\n'
        '{"audio": "a.wav", "text": "one"}\n'
        '\n'
        '{"audio": "b.wav", "text": "two"}\n',
        encoding="utf-8",
    )
    entries = bench._read_manifest(f)
    assert [e["text"] for e in entries] == ["one", "two"]
