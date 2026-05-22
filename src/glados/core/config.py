"""Config loader for `glados.toml` and `rooms.toml`.

Schema is intentionally tiny in v0 — fields will accrete as adapters land.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .protocols import Role


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    traces_dir: Path = Path("traces")


class AuthConfig(BaseModel):
    tokens: dict[str, str] = {}


class LLMConfig(BaseModel):
    backend: Literal["fake", "ollama"] = "fake"
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: float = 60.0


class AudioConfig(BaseModel):
    # Per-connection WAV trace of inbound mic audio. Useful for offline
    # replay against the STT; flip to false in production to stop
    # `traces/audio/` from filling at ~32 KB/s per active mic.
    wav_traces: bool = True


class VADConfig(BaseModel):
    # "fake" splits the stream into fixed-size utterances (dep-free,
    # used in tests). "silero" runs silero-vad on every 512-sample
    # chunk and emits real utterance boundaries.
    backend: Literal["fake", "silero"] = "silero"
    # Fake-only: how many int16 samples make up one utterance.
    # 16000 = 1 s at 16 kHz.
    fake_utterance_samples: int = 16000
    # Silero knobs. Threshold trades false-positives for missed speech;
    # min_silence_ms is how long quiet must last before we call the
    # utterance done; speech_pad_ms widens the emitted slice on each
    # side so Whisper sees a tiny breath of context.
    silero_threshold: float = 0.5
    silero_min_silence_ms: int = 200
    silero_speech_pad_ms: int = 30


class STTConfig(BaseModel):
    # "fake" returns `fake_text`. "faster-whisper" runs the configured
    # model; first call downloads the weights to HF_HOME.
    backend: Literal["fake", "faster-whisper"] = "faster-whisper"
    fake_text: str = "hello world"
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # None = auto-detect per utterance (multilingual). Pin to a code
    # (e.g. "en") only if you want to force one language.
    whisper_language: str | None = None


class TTSConfig(BaseModel):
    # "fake" yields a silent chunk (used in tests). "piper" loads a Piper
    # voice and streams real PCM. First piper construct downloads the
    # voice from HuggingFace into `voices_dir` if not already cached.
    backend: Literal["fake", "piper"] = "piper"
    piper_voice: str = "en_GB-cori-high"
    piper_voices_dir: Path = Path(r"E:\dev\piper\voices")


class GladosConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()
    llm: LLMConfig = LLMConfig()
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()


class ClientBinding(BaseModel):
    client_id: str
    room_id: str
    role: Role
    default_user: str = "default"


class RoomsConfig(BaseModel):
    clients: list[ClientBinding] = []

    def find(self, client_id: str) -> ClientBinding | None:
        return next((c for c in self.clients if c.client_id == client_id), None)


def load_glados_config(path: Path) -> GladosConfig:
    return _apply_env_overrides(GladosConfig(**_read_toml(path)))


def _apply_env_overrides(cfg: GladosConfig) -> GladosConfig:
    llm_updates: dict = {}
    backend = os.environ.get("GLADOS_LLM_BACKEND")
    if backend in ("fake", "ollama"):
        llm_updates["backend"] = backend
    if (model := os.environ.get("GLADOS_LLM_MODEL")) is not None:
        llm_updates["model"] = model
    if (host := os.environ.get("GLADOS_LLM_HOST")) is not None:
        llm_updates["host"] = host
    if llm_updates:
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update=llm_updates)})

    if (vad_backend := os.environ.get("GLADOS_VAD_BACKEND")) in ("fake", "silero"):
        cfg = cfg.model_copy(update={"vad": cfg.vad.model_copy(update={"backend": vad_backend})})

    if (stt_backend := os.environ.get("GLADOS_STT_BACKEND")) in ("fake", "faster-whisper"):
        cfg = cfg.model_copy(update={"stt": cfg.stt.model_copy(update={"backend": stt_backend})})

    if (tts_backend := os.environ.get("GLADOS_TTS_BACKEND")) in ("fake", "piper"):
        cfg = cfg.model_copy(update={"tts": cfg.tts.model_copy(update={"backend": tts_backend})})
    return cfg


def load_rooms_config(path: Path) -> RoomsConfig:
    return RoomsConfig(**_read_toml(path))


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)
