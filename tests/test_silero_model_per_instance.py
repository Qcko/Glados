"""Each SileroVAD owns its model: the model is stateful, so a shared one
lets two mic connections corrupt each other's speech detection."""

import sys
import types

from glados.audio.vad import silero


def _fake_silero_vad(monkeypatch) -> list[object]:
    loaded: list[object] = []

    def load_silero_vad():
        model = object()
        loaded.append(model)
        return model

    class VADIterator:
        def __init__(self, model, **_kwargs):
            self.model = model

    module = types.SimpleNamespace(load_silero_vad=load_silero_vad, VADIterator=VADIterator)
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    return loaded


def test_each_vad_gets_its_own_model(monkeypatch) -> None:
    loaded = _fake_silero_vad(monkeypatch)

    first = silero.SileroVAD()
    second = silero.SileroVAD()

    assert len(loaded) == 2
    assert first._vad.model is not second._vad.model
