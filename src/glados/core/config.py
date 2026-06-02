"""Config loader for `glados.toml` and `rooms.toml`.

Schema is intentionally tiny in v0 — fields will accrete as adapters land.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from .protocols import Role


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    traces_dir: Path = Path("traces")


class AuthConfig(BaseModel):
    # Client ids allowed to connect. Tokens themselves live in the OS
    # keyring under service `glados.client-tokens` (see core/secrets.py).
    clients: list[str] = []


class LLMConfig(BaseModel):
    backend: Literal["fake", "ollama"] = "fake"
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: float = 60.0


class RouterConfig(BaseModel):
    """v2.6 hybrid local/cloud router. Disabled by default — GLaDOS stays a
    pure local assistant until the operator opts in. `enabled` turns on
    per-turn routing; `cloud_enabled` is the separate, explicit privacy opt-in
    that permits tool args/results to cross to the cloud provider (ARCH §9).
    Both must be true AND an API key present for the cloud path to engage;
    otherwise every turn runs local."""

    enabled: bool = False
    cloud_enabled: bool = False
    # "anthropic": real cloud brain (needs cloud_enabled + an API key).
    # "local": loopback — the smart path runs on a local Ollama model too, so
    # the router, escalation, and UI can be exercised end-to-end before a cloud
    # provider is chosen. Nothing leaves the box, so it needs neither
    # cloud_enabled nor a key.
    provider: Literal["anthropic", "local"] = "anthropic"
    cloud_model: str = "claude-haiku-4-5-20251001"
    # provider="local" only: the Ollama tag for the smart path. Empty reuses
    # the same model as the local brain (true loopback — identical behaviour,
    # useful purely to see routing/escalation fire). Point it at a larger local
    # model (e.g. the 14b while the fast path runs a 7b) for a realistic split.
    local_smart_model: str = ""
    # API key handle: read from this env var at boot. Never stored in TOML
    # (ARCH §9 — TOML holds handles, not secrets). Absent key => cloud off.
    api_key_env: str = "ANTHROPIC_API_KEY"
    # Retry a `failed` local turn on cloud (router escalation input).
    escalate_on_failed: bool = True
    # Word count above which a request is treated as long/multi-clause and
    # routed to cloud by the deterministic rules.
    max_words_local: int = 30


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
    whisper_model: str = "distil-small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # Default "en" pins English-only decoding. Set to None for
    # auto-detect (paired with a multilingual model like `small`);
    # see configs/glados.toml for the full recipe.
    whisper_language: str | None = "en"


class TTSConfig(BaseModel):
    # "fake" yields a silent chunk (used in tests). "piper" loads a Piper
    # voice and streams real PCM. First piper construct downloads the
    # voice from HuggingFace into `voices_dir` if not already cached.
    backend: Literal["fake", "piper"] = "piper"
    piper_voice: str = "en_GB-cori-high"
    piper_voices_dir: Path = Path(
        os.environ.get("GLADOS_PIPER_VOICES_DIR")
        or (Path.home() / ".cache" / "piper" / "voices")
    )


class GladosConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()
    llm: LLMConfig = LLMConfig()
    router: RouterConfig = RouterConfig()
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()


class ToolOverlay(BaseModel):
    """GLaDOS-only flags applied on top of a tool spec fetched via real
    MCP `tools/list`. The MCP wire schema has no slot for trust/confirm
    flags — third-party servers don't know about them. We carry the
    flags in `servers.toml`, keyed by the tool's `name`, and overlay
    them after listing."""

    untrusted: bool = False
    requires_confirmation: bool = False
    # Marks a side-effecting tool (cart write, checkout) that is NOT gated by
    # confirmation, so the turn-outcome goal-check can still see that an action
    # landed. requires_confirmation already implies mutating; set this for
    # un-gated writes.
    mutating: bool = False
    timeout_s: float | None = None


class ServerEntry(BaseModel):
    id: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    autostart: bool = True
    # Origin gate for server-shipped memory (ARCH §14 layer 1). Only a
    # first-party server we vouch for is a candidate for trusted-prompt
    # injection of its `memory://lessons` resource. False (default) means
    # the lessons resource is never read into the system prompt, even if
    # the server exposes one. Origin trust ≠ content trust: a true flag
    # only makes the blob *eligible*; it still passes the LocalGuard
    # hash-approval gate before anything is injected.
    trusted: bool = False
    # Per-tool overlays keyed by the tool's `name` (not qualified).
    # Missing tools fall back to wire defaults (all flags False / None).
    tool_overlays: dict[str, ToolOverlay] = {}


class ServersConfig(BaseModel):
    server: list[ServerEntry] = []

    @model_validator(mode="after")
    def _unique_server_ids(self) -> "ServersConfig":
        seen: set[str] = set()
        for entry in self.server:
            if entry.id in seen:
                # Duplicate ids would silently overwrite each other's
                # tools in MCPRegistry (last writer wins), losing tools
                # without a peep. Fail loud at config-load time instead.
                raise ValueError(
                    f"duplicate server id in servers.toml: {entry.id!r}"
                )
            seen.add(entry.id)
        return self


def load_servers_config(path: Path) -> ServersConfig:
    return ServersConfig(**_read_toml(path))


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

    router_updates: dict = {}
    if (enabled := _env_bool("GLADOS_ROUTER_ENABLED")) is not None:
        router_updates["enabled"] = enabled
    if (cloud := _env_bool("GLADOS_ROUTER_CLOUD_ENABLED")) is not None:
        router_updates["cloud_enabled"] = cloud
    if (cloud_model := os.environ.get("GLADOS_ROUTER_CLOUD_MODEL")) is not None:
        router_updates["cloud_model"] = cloud_model
    if router_updates:
        cfg = cfg.model_copy(
            update={"router": cfg.router.model_copy(update=router_updates)}
        )
    return cfg


def _env_bool(name: str) -> bool | None:
    """Parse a boolean env var. Absent -> None (leave config default). Accepts
    1/true/yes/on (case-insensitive) as True, the rest as False."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_rooms_config(path: Path) -> RoomsConfig:
    return RoomsConfig(**_read_toml(path))


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)
