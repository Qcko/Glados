"""Config loader for `glados.toml` and `rooms.toml`.

Schema is intentionally tiny in v0 -- fields will accrete as adapters land.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from .protocols import Role

if TYPE_CHECKING:  # `adapters` is import-light, but keep config dependency-free at runtime
    from .adapters import ToolSpec


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    traces_dir: Path = Path("traces")
    # TLS (wss:// / https://). Both empty = plain HTTP. Self-signed cert pinned
    # by clients; generate with scripts/gen-tls-cert.sh. See deploy/ROADMAP.md.
    tls_certfile: str = ""
    tls_keyfile: str = ""
    # Loopback-only admin room-viewer port (observe any room's conversation as
    # text -- see ARCHITECTURE section 9). 0 = disabled. ALWAYS bound to 127.0.0.1 so
    # the house-wide capability never reaches the LAN, regardless of `host`.
    admin_port: int = 0


class HandshakeConfig(BaseModel):
    """Admission control for the /ws/v1 handshake (DoS + probe hardening;
    see client_room/deploy/DESIGN-ws-handshake-rate-limit.md).

    All controls key on the transport peer address -- no reverse proxy is
    assumed in front of GLaDOS. Behind one, every client would share the
    proxy's IP and the per-IP cap/lockout would misfire."""

    # Max seconds an accepted socket may take to complete the handshake.
    timeout_s: float = 10.0
    # Concurrent handshakes still pending auth: across all peers / per peer.
    max_pending: int = 8
    max_pending_per_ip: int = 2
    # Credential failures (unknown client id / bad token) within
    # `fail_window_s` that lock the source IP out for `lockout_s`. The
    # lockout self-expires and a successful handshake clears the counter.
    fail_threshold: int = 5
    fail_window_s: float = 60.0
    lockout_s: float = 60.0
    # Bound on the failure-tracking table (expired entries evict first).
    max_tracked_ips: int = 1024


class AuthConfig(BaseModel):
    # Client ids allowed to connect. Tokens themselves live in the OS
    # keyring under service `glados.client-tokens` (see core/secrets.py).
    clients: list[str] = []


class LLMConfig(BaseModel):
    backend: Literal["fake", "ollama"] = "fake"
    # The fallback a config lands on when it omits `model` -- deliberately the
    # SMALLEST capable model (4.18 GB), not the shipped one, so an accidental
    # default is cheap to load rather than a 6.6 GB surprise. Shipped model is
    # chosen in configs/glados.toml; keep this in sync with what is on disk --
    # a default naming an unpulled tag 404s at the first inference, not at boot.
    model: str = "qwen3:4b"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: float = 60.0
    # How long Ollama keeps the model resident after a request. "-1" pins it
    # in VRAM indefinitely so it never evicts mid-session -- an evicted model
    # re-colds, and a cold model drifts language / skips tools on the next
    # free-form turn (the boot warm-up only covers GLaDOS start). A duration
    # string ("30m", "1h") or "0" (evict immediately) are also accepted; passed
    # through verbatim to the /api/chat `keep_alive` field.
    keep_alive: str = "-1"
    # Context window sent as /api/chat options.num_ctx. Ollama otherwise applies
    # its own small default and truncates the prompt FROM THE FRONT, which
    # silently evicts the system prompt -- and with it the ARCH section 7
    # untrusted-content rule and the reply-language rule. Nothing else in GLaDOS
    # budgets tokens (history is capped by turn count, not size), so this is the
    # only ceiling there is. `None` sends no value and restores the old
    # behaviour: an escape hatch for a model whose default is already right.
    #
    # 8192 is deliberately conservative, NOT the model's maximum. Measured
    # 2026-08-17 on a 12 GB card: qwen2.5:14b-instruct-q5_K_M (retired
    # 19-08-2026) spilled to CPU at every context size (815 MiB at 4k, 1.7 GB at
    # 8k, 3.9 GB at 16k, 8.7 GB at 32k), and each spilled layer costs decode
    # speed. Raising this does not OOM -- it quietly offloads more and gets
    # slower, which is why the cost is easy to miss. Re-derive it per model and
    # per card; do not assume bigger. The qwen3 tags that replaced it both fit
    # the card whole at 8192 (zero spill), so headroom exists -- but note that
    # num_predict is now 4096, i.e. a third of this window can go on one reply.
    # This default (12288) is what a config OMITTING num_ctx gets -- it does NOT
    # fall through to Ollama's own default. The type allows None, but TOML has no
    # null, so None is reachable only by editing this line. Setting it too low is
    # not a soft failure: at 2048 with the full tool list the prompt truncates
    # from the FRONT and the model loses the tool definitions entirely (measured
    # 19-08-2026 -- 0/3 tool calls on every model tested, incl. qwen3).
    #
    # COUPLED TO num_predict. Generated tokens share this window, so keep
    # `max_assembled_prompt + num_predict <= num_ctx`. Raised 8192 -> 12288 on
    # 20-08-2026 because 4096 of num_predict against an ~4.7k steady-state
    # prompt exceeded 8192 by ~570 tokens. Both qwen3 tags still fit the 12 GB
    # card whole at 12288 (8b 7.23 GB, 4b 4.80 GB, zero spill).
    num_ctx: int | None = Field(default=12288, ge=1)
    # Generation cap sent as options.num_predict. Unbounded generation let an
    # observed repetition loop (the CallCheckLoginStatus leak, 2026-06-18) run
    # until it filled the context; this bounds that failure class to one turn.
    # Must stay comfortably above the longest legitimate spoken reply.
    #
    # MODEL-DEPENDENT -- re-derive on every model swap. A reasoning model spends
    # this budget inside its <think> block before emitting anything, so a cap
    # sized for a non-reasoning model truncates it mid-reasoning and the turn
    # surfaces as an EMPTY reply with no tool call. Measured 2026-08-19 on the
    # T1-T12 suite: at 512, qwen3:4b scored 4/22 and qwen3:8b 18/22; at 4096 the
    # same models scored 20/22 and 21/22. 512 was ample for the non-reasoning
    # qwen2.5 incumbent, which is exactly why the old "model-independent" note
    # here was both wrong and hard to doubt.
    #
    # The cost of raising it is that the repetition loop above is bounded 8x
    # looser. Accepted deliberately: a wedged turn is recoverable, a silently
    # blanked assistant is not.
    num_predict: int | None = Field(default=4096, ge=1)
    # Sent as /api/chat `think`. `None` omits the key and leaves the model's own
    # default alone; `False` asks a reasoning model not to reason.
    #
    # STRICTLY PER-MODEL -- do NOT promote this to a global default. Measured
    # 2026-08-25 on the same request: qwen3:8b honours it properly (tool-pick
    # 2.4s -> 0.4s, final answer 7.8s -> 0.9s, no reasoning emitted anywhere),
    # while qwen3:4b does NOT suppress the reasoning at all -- it relocates it
    # into `content`, which is the SPOKEN channel, stray "</think>" tag
    # included. Turning this on for 4b makes GLaDOS read its own chain of
    # thought aloud.
    #
    # Qwen's own "/no_think" prompt marker is not an alternative: Ollama's chat
    # template overrides it and the model reasons anyway (measured, both tags).
    think: bool | None = None
    # Language GLaDOS must reply in. The language guard (core/language_guard.py)
    # rewrites a free-form reply whose dominant script is not this language's --
    # a deterministic backstop for the cold-model drift the prompt rule alone
    # does not stop. Only mapped languages are guarded; others fail open.
    reply_language: str = "en"
    # Persona/verbosity override for the static system prompt. Empty (default)
    # uses the built-in SYSTEM_PROMPT shipped in brain/prompts/system.py; set a
    # full replacement here to retune persona per demo without editing code.
    # Hash-approved server memory (ARCH section 14) is still appended on top of
    # whichever base wins.
    system_prompt: str = ""


class RouterConfig(BaseModel):
    """v2.6 local multi-model router. Disabled by default -- GLaDOS runs every
    turn on the single primary brain until the operator opts in. `enabled` turns
    on per-turn routing between the primary brain and a specialist. The
    specialist is local by default (`provider="local"`); the `cloud_*` knobs
    gate the dormant cloud escape hatch and stay off unless explicitly opted in
    (ARCH section 12)."""

    enabled: bool = False
    # The cloud escape hatch is off by default and gated separately from
    # `enabled`: it is the explicit opt-in that permits tool args/results to
    # cross to the external provider (ARCH section 9). `provider="anthropic"` engages
    # only when this is true AND an API key is present.
    cloud_enabled: bool = False
    # "local": the default all-local specialist -- runs on a local Ollama model,
    # nothing leaves the box, needs neither cloud_enabled nor a key.
    # "anthropic": the dormant cloud escape hatch (needs cloud_enabled + an API
    # key). Endpoint hardcoded to api.anthropic.com; see ARCH section 12.
    provider: Literal["anthropic", "local"] = "local"
    cloud_model: str = "claude-haiku-4-5-20251001"
    # provider="local" only: the Ollama tag for the specialist. Empty reuses the
    # same model as the primary brain (alias -- identical behaviour, useful purely
    # to see routing/escalation fire). Point it at a larger local model (e.g. a
    # 14b while the primary runs a 7b) for a realistic split.
    local_smart_model: str = ""
    # API key handle: read from this env var at boot. Never stored in TOML
    # (ARCH section 9 -- TOML holds handles, not secrets). Absent key => cloud off.
    api_key_env: str = "ANTHROPIC_API_KEY"
    # Retry a `failed` primary turn on the specialist (router escalation input).
    escalate_on_failed: bool = True
    # Word count above which a request is treated as long/multi-clause and
    # routed to the specialist by the deterministic rules.
    max_words_local: int = 30


class SessionConfig(BaseModel):
    """Conversation continuity (ARCH section 3 idle-window, section 8 hot ring buffer).

    A follow-up utterance in the same `(room_id, speaker_id)` reuses the live
    session -- and its replayed history -- when it arrives within
    `idle_window_s` of the last activity; after that gap a fresh session opens
    with empty history. `history_max_turns` bounds how many prior turns (a turn
    = the user message plus the assistant/tool messages it produced) are
    replayed into the prompt."""

    idle_window_s: float = 180.0
    history_max_turns: int = 8


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
    # These ARE the end-of-utterance hangover and the pre-roll -- both already
    # existed, they were just set too tight. Retuned 20-08-2026 offline (Piper
    # speech, silence-trimmed, spliced gap, fed through the real SileroVAD).
    # min_silence 200 split a sentence on a 300 ms gap; a breath is 300-600 ms.
    # 500 holds gaps to 400 ms. It is NOT free -- it adds ~300 ms before GLaDOS
    # starts replying, on every turn.
    # speech_pad is a HEDGE, not a measured fix: on clean speech the old 30 ms
    # already lost nothing, so the dropped-onset symptom did not reproduce here
    # and most likely lives in the capture/AEC path, which no VAD value can fix.
    # Raising it is free (SileroVAD holds every chunk for up to 60 s anyway).
    silero_min_silence_ms: int = 800
    silero_speech_pad_ms: int = 200


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
    # Vocabulary biasing -- no model retraining. Nudges decoding toward domain
    # words/names Whisper otherwise mangles. `whisper_hotwords` is a
    # space-separated boost list; `whisper_initial_prompt` is freeform context
    # that can also fix spelling. Both empty = no biasing (unchanged behaviour).
    # Grow them from logged mis-transcriptions -- the pragmatic "make STT learn"
    # path short of fine-tuning (which only pays off once v5 speaker-ID yields
    # labelled per-speaker audio).
    whisper_hotwords: str = ""
    whisper_initial_prompt: str = ""


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
    # Pronunciation fixes applied to the TTS text only (the chat surface keeps
    # the original wording). Piper is phoneme-based and mangles proper
    # nouns/acronyms; map the written form -> a phonetic spelling. Whole-word,
    # case-insensitive. Add a row per mispronounced name.
    pronunciations: dict[str, str] = {"GLaDOS": "Gladoss"}
    # Feedback-gate timing (server-side mic-mute that stops a room's speaker
    # output from self-triggering a turn via its own mic -- see Organizer). The
    # gate holds a room's mic for the *estimated playback duration* of the reply
    # (so it scales with reply length, unlike a fixed cooldown), then a short
    # tail cooldown. A speaker client may shorten the estimate by reporting
    # PlaybackDone. Bump these for high-latency sinks (Bluetooth, external
    # speakers); a room device with no echo cancellation needs the margin.
    gate_cooldown_s: float = 0.200      # reverb/tail after playback is judged done
    gate_drain_margin_s: float = 0.5    # padding on the duration estimate (jitter buffer + sink latency)
    gate_max_s: float = 120.0           # hard ceiling on a single gate (guards a duration-accounting bug)


class GladosConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    handshake: HandshakeConfig = HandshakeConfig()
    auth: AuthConfig = AuthConfig()
    llm: LLMConfig = LLMConfig()
    router: RouterConfig = RouterConfig()
    session: SessionConfig = SessionConfig()
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()


class ToolOverlay(BaseModel):
    """GLaDOS-only flags applied on top of a tool spec fetched via real
    MCP `tools/list`. The MCP wire schema has no slot for trust/confirm
    flags -- third-party servers don't know about them. We carry the
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
    # Spoken-length cap on the result (core/tool_payload_cap.py). `max_items`
    # is how many reach the model; `flex_to` speaks the whole list up to that
    # size rather than withholding a trivial remainder; `items_key` names the
    # key holding the list when the payload is an object rather than a bare
    # array. Structural only -- no server's field names live here, and the
    # ORDER is the server's to decide, since it owns the data it ranked.
    max_items: int | None = None
    flex_to: int | None = None
    items_key: str | None = None


class ServerEntry(BaseModel):
    id: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    autostart: bool = True
    # Lazy spawn + idle reap (ARCH section 13). When True the server is still
    # spawned at startup to list its tools (so the LLM sees them), then put
    # dormant immediately; the first tool dispatch wakes it, and the reaper
    # sleeps it again after `idle_timeout_s` of no calls. Default False keeps
    # the eager behaviour -- the child stays resident for the whole session.
    # Flip Dunnes to lazy once stable so Selenium/Chrome isn't held when no
    # one is shopping. Ignored when `autostart` is False (nothing to spawn).
    lazy: bool = False
    idle_timeout_s: float = 300.0
    # Origin gate for server-shipped memory (ARCH section 14 layer 1). Only a
    # first-party server we vouch for is a candidate for trusted-prompt
    # injection of its `memory://lessons` resource. False (default) means
    # the lessons resource is never read into the system prompt, even if
    # the server exposes one. Origin trust != content trust: a true flag
    # only makes the blob *eligible*; it still passes the LocalGuard
    # hash-approval gate before anything is injected.
    trusted: bool = False
    # Server-level untrusted FLOOR (ARCH section 7). When true, every tool this
    # server exposes returns content from outside the local trust boundary,
    # so the Organizer wraps every result in <external> -- including tools
    # added by a future version of that server which nobody thought to list
    # here.
    #
    # This exists because the per-tool flag is fail-OPEN: a server that grows
    # a tool has it treated as trusted until a human remembers to add an
    # overlay, and "remembers" is not a security control. A whole-server
    # judgement is also the one an operator can actually make correctly --
    # "everything Dunnes returns is a live scrape", "anything the calendar
    # returns may have been authored from a phone".
    #
    # A FLOOR, not a default: an overlay may raise one tool to untrusted, but
    # cannot lower one below the server's setting. Marking content *trusted*
    # is precisely the direction that must never happen by accident.
    untrusted: bool = False
    # Per-tool overlays keyed by the tool's `name` (not qualified).
    # Missing tools fall back to wire defaults (all flags False / None).
    tool_overlays: dict[str, ToolOverlay] = {}
    # Tiered tool-scoping (ARCH section 13). Words that route a turn to THIS server's
    # tools -- same whole-word keyword style as brain/router/rules.py. When set,
    # the server's tools are offered to the model ONLY on turns whose text
    # matches (so a big server like Dunnes can't swamp the ~30-tool ceiling on
    # unrelated turns -- the project-callcheck-tooltext leak). Empty = unscoped:
    # the server's tools are always offered (back-compat default).
    intent_keywords: list[str] = []
    # Always offer this server's tools regardless of intent (the "core tools
    # always on" allowlist, e.g. a time server). Overrides intent_keywords.
    core: bool = False

    def apply_flags(self, spec: "ToolSpec") -> "ToolSpec":
        """Return `spec` with the GLaDOS-only flags this config declares.

        Real MCP has no slot for trust/confirm flags, so they are merged here
        rather than read off the wire.

        `untrusted` is OR-ed with the server floor; every other flag comes
        from the overlay alone. The asymmetry is the point: an overlay that
        sets only `timeout_s` still constructs a full `ToolOverlay`, whose
        other fields default to False -- so a plain assignment would let a
        timeout tweak quietly un-mark a tool as untrusted, which is a
        security downgrade written as a performance edit.
        """
        overlay = self.tool_overlays.get(spec.name) or ToolOverlay()
        return spec.model_copy(
            update={
                "untrusted": self.untrusted or overlay.untrusted,
                "requires_confirmation": overlay.requires_confirmation,
                "mutating": overlay.mutating,
                "timeout_s": overlay.timeout_s,
                "max_items": overlay.max_items,
                "flex_to": overlay.flex_to,
                "items_key": overlay.items_key,
            }
        )


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
    server_updates: dict = {}
    if (host := os.environ.get("GLADOS_HOST")) is not None:
        server_updates["host"] = host
    if (port_str := os.environ.get("GLADOS_PORT")) is not None:
        try:
            server_updates["port"] = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"GLADOS_PORT must be an integer, got {port_str!r}"
            ) from exc
    if (admin_str := os.environ.get("GLADOS_ADMIN_PORT")) is not None:
        try:
            server_updates["admin_port"] = int(admin_str)
        except ValueError as exc:
            raise ValueError(
                f"GLADOS_ADMIN_PORT must be an integer, got {admin_str!r}"
            ) from exc
    if (cert := os.environ.get("GLADOS_TLS_CERT")) is not None:
        server_updates["tls_certfile"] = cert
    if (key := os.environ.get("GLADOS_TLS_KEY")) is not None:
        server_updates["tls_keyfile"] = key
    if server_updates:
        cfg = cfg.model_copy(update={"server": cfg.server.model_copy(update=server_updates)})

    handshake_updates: dict = {}
    for env_name, field_name, parse in (
        ("GLADOS_HANDSHAKE_TIMEOUT_S", "timeout_s", float),
        ("GLADOS_HANDSHAKE_MAX_PENDING", "max_pending", int),
        ("GLADOS_HANDSHAKE_MAX_PENDING_PER_IP", "max_pending_per_ip", int),
        ("GLADOS_HANDSHAKE_FAIL_THRESHOLD", "fail_threshold", int),
        ("GLADOS_HANDSHAKE_FAIL_WINDOW_S", "fail_window_s", float),
        ("GLADOS_HANDSHAKE_LOCKOUT_S", "lockout_s", float),
        ("GLADOS_HANDSHAKE_MAX_TRACKED_IPS", "max_tracked_ips", int),
    ):
        if (raw := os.environ.get(env_name)) is not None:
            try:
                handshake_updates[field_name] = parse(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{env_name} must be a {parse.__name__}, got {raw!r}"
                ) from exc
    if handshake_updates:
        cfg = cfg.model_copy(
            update={"handshake": cfg.handshake.model_copy(update=handshake_updates)}
        )

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
