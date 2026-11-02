"""ONNX speaker-id backends, WeSpeaker and ECAPA (DESIGN-speaker-id.md, slice 3).

The model files are not in the repo; the tests that need one run only when
`GLADOS_VOICEID_MODEL` (WeSpeaker) or `GLADOS_VOICEID_ECAPA_MODEL` points at it.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from glados.core.config import SpeakerIdConfig
from glados.core.secrets import InMemorySecrets
from glados.voiceid import bakeoff, build_identifier
from glados.voiceid.embedder import SAMPLE_RATE, FakeEmbedder, _fbank, unit
from glados.voiceid.identifier import decide
from glados.voiceid.store import Voiceprint

MODEL = os.environ.get("GLADOS_VOICEID_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_file(), reason="GLADOS_VOICEID_MODEL not set")
ECAPA = os.environ.get("GLADOS_VOICEID_ECAPA_MODEL", "")
needs_ecapa = pytest.mark.skipif(
    not ECAPA or not Path(ECAPA).is_file(), reason="GLADOS_VOICEID_ECAPA_MODEL not set"
)


def _tone(hz: float, seconds: float = 2.0) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * hz * t) * 8000).astype(np.int16).tobytes()


def test_fbank_is_80_bins_at_10_ms_with_the_mean_removed():
    feats = _fbank(_tone(220.0))
    assert feats.shape[1] == 80 and abs(feats.shape[0] - 198) <= 2
    assert feats.dtype == np.float32
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-4)


def test_a_missing_model_degrades_to_unidentified_not_a_failed_boot(tmp_path):
    cfg = SpeakerIdConfig(backend="wespeaker", model_path=tmp_path / "absent.onnx", store_dir=tmp_path)
    assert build_identifier(cfg, InMemorySecrets()) is None


@pytest.mark.parametrize("backend", ["wespeaker", "ecapa"])
def test_each_onnx_backend_degrades_on_a_missing_model(tmp_path, backend):
    cfg = SpeakerIdConfig(backend=backend, model_path=tmp_path / "absent.onnx", store_dir=tmp_path)
    assert build_identifier(cfg, InMemorySecrets()) is None


def test_threads_are_capped_at_two():
    with pytest.raises(ValueError):
        SpeakerIdConfig(threads=3)


@needs_model
def test_embedding_is_a_unit_256_vector_and_model_id_names_the_file():
    from glados.voiceid.embedder import WeSpeakerEmbedder

    embedder = WeSpeakerEmbedder(Path(MODEL))
    vector = embedder.embed(_tone(180.0))
    assert vector.shape == (256,) and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    assert embedder.model_id.startswith("wespeaker-resnet34-LM:") and len(embedder.model_id.split(":")[1]) == 12


@needs_model
def test_the_same_clip_embeds_identically():
    from glados.voiceid.embedder import WeSpeakerEmbedder

    embedder = WeSpeakerEmbedder(Path(MODEL))
    clip = _tone(300.0)
    assert float(embedder.embed(clip) @ embedder.embed(clip)) > 0.999


@needs_ecapa
def test_ecapa_embeds_a_unit_192_vector_and_names_its_model():
    from glados.voiceid.embedder import EcapaEmbedder

    embedder = EcapaEmbedder(Path(ECAPA))
    vector = embedder.embed(_tone(180.0))
    assert vector.shape == (192,) and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    assert embedder.model_id.startswith("ecapa-speechbrain-voxceleb:")


@needs_ecapa
def test_ecapa_cuts_a_long_clip_to_its_first_30_seconds():
    from glados.voiceid.embedder import EcapaEmbedder

    embedder = EcapaEmbedder(Path(ECAPA))
    head = np.random.default_rng(3).integers(-3000, 3000, 30 * SAMPLE_RATE, dtype=np.int16).tobytes()
    tail = _tone(440.0, seconds=10.0)
    assert float(embedder.embed(head + tail) @ embedder.embed(head)) > 0.9999


@needs_ecapa
def test_ecapa_refuses_under_a_second_so_the_turn_goes_unidentified():
    from glados.voiceid.embedder import EcapaEmbedder

    with pytest.raises(ValueError, match="1 s"):
        EcapaEmbedder(Path(ECAPA)).embed(_tone(200.0, seconds=0.5))


@needs_model
@needs_ecapa
def test_each_backend_refuses_the_other_backends_model_file(tmp_path):
    from glados.voiceid.embedder import EcapaEmbedder, WeSpeakerEmbedder

    with pytest.raises(ValueError, match="feeds 'wav'"):
        EcapaEmbedder(Path(MODEL))
    with pytest.raises(ValueError, match="feeds 'feats'"):
        WeSpeakerEmbedder(Path(ECAPA))
    swapped = SpeakerIdConfig(backend="ecapa", model_path=Path(MODEL), store_dir=tmp_path)
    assert build_identifier(swapped, InMemorySecrets()) is None


# ---- threshold bake-off -----------------------------------------------------


def _clip(seed: int) -> bytes:
    return np.random.default_rng(seed).integers(-3000, 3000, SAMPLE_RATE, dtype=np.int16).tobytes()


def _write(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def _voice(base: np.ndarray, seed: int, spread: float = 0.15) -> np.ndarray:
    return base + spread * np.random.default_rng(seed).standard_normal(base.shape)


@pytest.fixture
def household(tmp_path):
    rng = np.random.default_rng(0)
    alice, bob = rng.standard_normal(16), rng.standard_normal(16)
    voices = {}
    for offset, (person, base) in enumerate((("alice", alice), ("bob", bob))):
        for i in range(4):
            pcm = _clip(100 * offset + i)
            voices[pcm] = _voice(base, 100 * offset + i)
            _write(tmp_path / person / f"{i}.wav", pcm)
    for i in range(3):
        _write(tmp_path / "impostor" / f"tv-{i}.wav", _clip(50_000 + i))
    return tmp_path, FakeEmbedder(voices, dim=16)


def test_bakeoff_accepts_every_genuine_clip_and_flags_thin_impostor_evidence(household):
    clip_dir, embedder = household
    data = bakeoff.run(clip_dir, embedder, margin=0.10)
    assert data["people"] == ["alice", "bob"] and data["impostor_groups"] == {"tv": 3}
    chosen = data["pick"]
    assert chosen["genuine_accept"] == 1.0
    assert chosen["impostor_accept"] == 0.0 and chosen["wrong_person"] == 0.0
    assert data["enough_impostors"] is False
    assert "PROVISIONAL" in bakeoff.headline(data)


def _prints(**centroids) -> tuple[Voiceprint, ...]:
    return tuple(Voiceprint(name, "m", unit(c)) for name, c in centroids.items())


def test_a_genuine_clip_taken_for_someone_else_is_a_wrong_person_not_an_impostor():
    prints = _prints(alice=[1.0, 0.0], bob=[0.0, 1.0])
    probes = [bakeoff.Probe("alice", "genuine", unit([0.1, 1.0]), prints)]
    point = next(p for p in bakeoff.sweep(probes, margin=0.1) if p.threshold == 0.5)
    assert point.wrong_persons == 1 and point.wrong_person == 1.0
    assert point.impostor_accept == 0.0 and point.genuine_accept == 0.0
    assert bakeoff.pick([point]) is None


def test_with_no_impostors_the_impostor_rate_is_zero_and_evidence_is_thin(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.standard_normal(8)
    voices = {}
    for i in range(3):
        pcm = _clip(900 + i)
        voices[pcm] = _voice(base, 900 + i)
        _write(tmp_path / "alice" / f"{i}.wav", pcm)
    data = bakeoff.run(tmp_path, FakeEmbedder(voices, dim=8), margin=0.1)
    assert data["impostor_groups"] == {} and data["enough_impostors"] is False
    assert all(p["impostor_accept"] == 0.0 for p in data["sweep"])


def test_bakeoff_agrees_with_the_identifiers_own_decide():
    rng = np.random.default_rng(7)
    prints = _prints(alice=rng.standard_normal(8), bob=rng.standard_normal(8))
    for seed in range(50):
        probe = unit(np.random.default_rng(seed).standard_normal(8))
        expected = decide(probe, prints, accept_threshold=0.4, margin=0.1).person
        assert bakeoff._decide(bakeoff.Probe(None, "tv", probe, prints), 0.4, 0.1) == expected


def test_pick_is_the_lowest_threshold_inside_the_gate():
    points = [
        bakeoff.Point(0.3, 1.0, 0.2, 0.0, 2, 0),
        bakeoff.Point(0.4, 0.9, 0.0, 0.05, 0, 1),
        bakeoff.Point(0.5, 0.8, 0.0, 0.0, 0, 0),
    ]
    assert bakeoff.pick(points).threshold == 0.5


def test_bakeoff_refuses_a_person_with_one_clip(tmp_path):
    _write(tmp_path / "alice" / "0.wav", _clip(1))
    with pytest.raises(ValueError, match=r"2\+ clips"):
        bakeoff.run(tmp_path, FakeEmbedder(), margin=0.1)


def test_bakeoff_report_holds_no_vectors(household, tmp_path):
    clip_dir, embedder = household
    path = bakeoff.write_report(bakeoff.run(clip_dir, embedder, 0.1), tmp_path / "out", now=lambda: "t")
    text = path.read_text(encoding="utf-8")
    assert path.name == "voiceid_bakeoff_t.json" and "embedding" not in text and len(text) < 20_000
