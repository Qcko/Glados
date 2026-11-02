"""Speaker-embedding backends behind one small Protocol.

`embed` is synchronous and CPU-bound; the identifier runs it on its own
worker thread. A returned vector is biometric data (DESIGN-speaker-id.md):
it is compared and dropped, never logged or traced.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000


class SpeakerEmbedder(Protocol):
    model_id: str

    def embed(self, pcm: bytes) -> np.ndarray: ...


def unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero embedding")
    return v / norm


class FakeEmbedder:
    """Test backend. Known pcm maps to a preset voice; anything else gets a
    stable pseudo-random vector derived from its bytes, so two different
    clips are near-orthogonal strangers."""

    def __init__(
        self,
        voices: Mapping[bytes, Sequence[float]] | None = None,
        *,
        dim: int = 8,
        delay_s: float = 0.0,
        error: Exception | None = None,
        model_id: str = "fake-v1",
    ) -> None:
        self.model_id = model_id
        self._voices = dict(voices or {})
        self._dim = dim
        self._delay_s = delay_s
        self._error = error
        self.calls = 0

    def embed(self, pcm: bytes) -> np.ndarray:
        self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        if pcm in self._voices:
            return unit(self._voices[pcm])
        return unit(self._stranger(pcm))

    def _stranger(self, pcm: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(pcm).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self._dim)


class WeSpeakerEmbedder:
    """WeSpeaker ResNet34 exported to ONNX, on CPU. Its frontend is 80-bin
    Kaldi fbank (25 ms / 10 ms, no dither) with per-utterance mean removal,
    matching how the model was trained. `model_id` carries a hash of the
    model file, so a re-export invalidates every enrolled print.

    `threads` caps the ONNX pass only. The fbank runs on torch's process-wide
    pool, shared with Silero VAD, and is left alone: it costs about 2 ms of a
    ~150 ms embed, so pinning torch globally would cost the VAD more than it
    saves here."""

    _FAMILY = "wespeaker-resnet34-LM"
    _INPUT = "feats"

    def __init__(self, model_path: Path, *, threads: int = 1) -> None:
        self._session = _cpu_session(model_path, threads, self._INPUT)
        self.model_id = f"{self._FAMILY}:{_file_digest(model_path)}"
        _warm_up(self)

    def embed(self, pcm: bytes) -> np.ndarray:
        feats = _fbank(pcm)
        [embedding] = self._session.run(None, {self._INPUT: feats[np.newaxis]})
        return unit(embedding[0])


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) exported to one ONNX
    graph with its Fbank and sentence normalisation inside, so it needs no
    torch: waveform in [-1, 1] in, 192-dim embedding out. Export recipe and
    parity check: `<lab-root>/glados/ecapa_export/`.

    The export declared 1-30 s clips. Under 1 s is refused, so the turn goes
    unidentified rather than matching on a vector the graph was never checked
    for. Over 30 s keeps the first 30 s: the onset is the person addressing
    GLaDOS, and a long capture's tail is likelier to be TV or someone else."""

    _FAMILY = "ecapa-speechbrain-voxceleb"
    _INPUT = "wav"
    MIN_SAMPLES = SAMPLE_RATE
    MAX_SAMPLES = 30 * SAMPLE_RATE

    def __init__(self, model_path: Path, *, threads: int = 1) -> None:
        self._session = _cpu_session(model_path, threads, self._INPUT)
        self.model_id = f"{self._FAMILY}:{_file_digest(model_path)}"
        _warm_up(self)

    def embed(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm, dtype=np.int16)[: self.MAX_SAMPLES]
        if len(samples) < self.MIN_SAMPLES:
            raise ValueError(f"ecapa needs 1 s of audio, got {len(samples) / SAMPLE_RATE:.2f} s")
        wav = samples.astype(np.float32) / 32768.0
        [embedding] = self._session.run(None, {self._INPUT: wav[np.newaxis]})
        return unit(embedding[0])


def _warm_up(embedder: SpeakerEmbedder) -> None:
    """The first embed pays session init (and for WeSpeaker the torch
    import), about 2.6 s on the devbox: well past the turn's identify cap.
    Pay it at load time instead, on a second of silence."""
    embedder.embed(bytes(2 * SAMPLE_RATE))


def _cpu_session(model_path: Path, threads: int, expected_input: str):
    """Refuses a model file whose input is not the one this backend feeds,
    so `backend` and `model_path` pointing at different models fail at load
    (degrading to unidentified) instead of embedding garbage."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    actual = session.get_inputs()[0].name
    if actual != expected_input:
        raise ValueError(f"{Path(model_path).name} takes input {actual!r}, this backend feeds {expected_input!r}")
    return session


def _fbank(pcm: bytes) -> np.ndarray:
    import torch
    import torchaudio.compliance.kaldi as kaldi

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    feats = kaldi.fbank(
        torch.from_numpy(samples).unsqueeze(0),
        num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
        sample_frequency=SAMPLE_RATE, window_type="hamming", use_energy=False,
    )
    return (feats - feats.mean(dim=0)).numpy().astype(np.float32)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]
