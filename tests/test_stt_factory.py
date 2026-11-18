"""STT factory: CUDA with a checked warm-up, else the small CPU model."""

from __future__ import annotations

import pytest

from glados.core import server
from glados.core.config import STTConfig


class _Built:
    def __init__(self, model: str, device: str, compute_type: str, fail: bool) -> None:
        self.model, self.device, self.compute_type, self._fail = model, device, compute_type, fail

    def check(self) -> None:
        if self._fail:
            raise RuntimeError("cublas64_12.dll not found")


def _patch(monkeypatch, *, cuda_fails: bool) -> list[_Built]:
    built: list[_Built] = []

    def fake(cfg, model, device, compute_type):
        stt = _Built(model, device, compute_type, fail=cuda_fails and device == "cuda")
        built.append(stt)
        return stt

    monkeypatch.setattr(server, "_build_whisper", fake)
    return built


def test_cuda_failure_falls_back_to_small_cpu_model(monkeypatch, caplog) -> None:
    built = _patch(monkeypatch, cuda_fails=True)
    cfg = STTConfig(whisper_model="large-v3-turbo", whisper_device="cuda",
                    whisper_compute_type="float16")
    stt = server._build_stt(cfg)
    assert (stt.model, stt.device, stt.compute_type) == ("distil-small.en", "cpu", "int8")
    assert [b.device for b in built] == ["cuda", "cpu"]
    assert "falling back" in caplog.text


def test_cuda_success_keeps_cuda(monkeypatch) -> None:
    _patch(monkeypatch, cuda_fails=False)
    stt = server._build_stt(STTConfig(whisper_device="cuda", whisper_compute_type="float16"))
    assert stt.device == "cuda"


def test_cpu_config_builds_once_without_check(monkeypatch) -> None:
    built = _patch(monkeypatch, cuda_fails=True)
    stt = server._build_stt(STTConfig())
    assert stt.device == "cpu" and len(built) == 1
