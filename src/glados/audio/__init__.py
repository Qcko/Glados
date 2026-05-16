"""Audio adapters: wake-word, VAD, STT, TTS.

Real backends live in subpackages (vad/, stt/, ...) selected by config.
Fakes alongside, used by tests and by the default `backend = "fake"`
config so the server boots without ML deps."""
