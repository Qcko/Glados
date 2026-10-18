"""openWakeWord backend for the `WakeWord` adapter.

The model files (the wake model plus openWakeWord's shared melspectrogram and
embedding models) are loaded from a local directory and never downloaded at
runtime: `verify_model_files` checks each against a pinned sha256 at boot, and
a missing or mismatched file refuses the boot (DESIGN.md decision 11).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# openWakeWord scores 80 ms frames; feeding it whole frames keeps its
# internal streaming buffers aligned.
_FRAME_SAMPLES = 1280
_FRAME_BYTES = _FRAME_SAMPLES * 2


class WakeModelError(RuntimeError):
    pass


def verify_model_files(model_dir: Path, pinned: dict[str, str]) -> None:
    """Refuse unless every pinned file exists and hashes to its pin."""
    if not pinned:
        raise WakeModelError("wake: no model sha256 pins configured ([wake] sha256)")
    for name, expected in pinned.items():
        path = model_dir / name
        if not path.is_file():
            raise WakeModelError(f"wake: model file {name} missing from the wake model dir")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise WakeModelError(f"wake: model file {name} sha256 mismatch (got {actual})")


class OpenWakeWord:
    def __init__(
        self,
        *,
        model_path: Path,
        melspec_path: Path,
        embedding_path: Path,
        threshold: float,
        model_factory=None,
    ) -> None:
        self._threshold = threshold
        self._model = (model_factory or _load_model)(model_path, melspec_path, embedding_path)
        self._buf = bytearray()

    def feed(self, pcm: bytes) -> list[float]:
        self._buf.extend(pcm)
        scores: list[float] = []
        while len(self._buf) >= _FRAME_BYTES:
            frame = np.frombuffer(bytes(self._buf[:_FRAME_BYTES]), dtype="<i2")
            del self._buf[:_FRAME_BYTES]
            best = max(self._model.predict(frame).values(), default=0.0)
            if best >= self._threshold:
                scores.append(float(best))
        return scores

    def reset(self) -> None:
        self._buf.clear()
        self._model.reset()


def _load_model(model_path: Path, melspec_path: Path, embedding_path: Path):
    from openwakeword.model import Model

    return Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(melspec_path),
        embedding_model_path=str(embedding_path),
    )
