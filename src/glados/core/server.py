"""FastAPI server exposing `/ws/v1`.

Handshake → register client connection → forward `user_text` to the
Organizer. The Organizer owns sessions, tool dispatch, and egress; this file
only handles wire I/O and connection bookkeeping.

Construction is done by `build_app(config_dir)` — a factory that wires
config, components, and the Organizer into a fresh `FastAPI` instance with
all runtime state on `app.state`. Tests can build isolated apps; production
imports the module-level `app = build_app()` default.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..audio.pipeline import AudioPipeline
from ..audio.stt.fake import FakeSTT
from ..audio.tts.fake import FakeTTS
from ..audio.vad.fake import FakeVAD
from ..brain.llm.fake import FakeLLM
from ..brain.llm.ollama import OllamaLLM
from ..brain.router import Router
from ..mcp.registry import MCPRegistry
from ..mcp.stdio_client import StdioServer, StdioToolProxy
from ..servers.time_server import NowTool
from ..servers.toy_server import TOY_TOOLS
from .adapters import LLM, STT, TTS, VAD
from .audio_sink import AudioSink, FrameTooShort
from .config import (
    GladosConfig,
    LLMConfig,
    RoomsConfig,
    RouterConfig,
    ServersConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    load_glados_config,
    load_rooms_config,
    load_servers_config,
)
from .handshake_gate import HandshakeGate, Verdict
from .logging_setup import setup_logging
from . import memory_gate
from .ollama_lifecycle import OllamaLifecycle
from .organizer import Organizer
from .secrets import KeyringSecrets, SecretsStore
from .protocols import (
    ClientMessage,
    ErrorMessage,
    Hello,
    Interrupt,
    MemoryBlockNotice,
    ToolConfirmResponse,
    UserText,
)
from .sessions import SessionRegistry
from .traces import TraceStore


log = logging.getLogger(__name__)

CONFIG_DIR_ENV = "GLADOS_CONFIG_DIR"

# How often the idle reaper sweeps lazy MCP servers (ARCH §13). Well under any
# sane `idle_timeout_s` so a server is reaped within ~one tick of going idle.
_LAZY_REAPER_TICK_S = 30.0


def _reap_idle_servers(
    lazy_servers: list[tuple["StdioServer", float]], now: float
) -> list[Awaitable[None]]:
    """Return a sleep() coroutine for every resident lazy server idle past its
    timeout. Pure of timing — the caller supplies `now` and awaits the results,
    which keeps it unit-testable with a fake clock."""
    return [
        srv.sleep()
        for srv, idle_timeout_s in lazy_servers
        if srv.is_resident() and srv.idle_seconds(now) >= idle_timeout_s
    ]


async def _lazy_reaper(lazy_servers: list[tuple["StdioServer", float]]) -> None:
    """Periodically sleep lazy MCP servers that have gone idle. Runs as a
    lifespan-owned background task; cancelled on shutdown."""
    while True:
        await asyncio.sleep(_LAZY_REAPER_TICK_S)
        now = asyncio.get_running_loop().time()
        for coro in _reap_idle_servers(lazy_servers, now):
            await coro


def _resolve_servers_toml(cfg_dir: Path) -> Path:
    real = cfg_dir / "servers.toml"
    if real.exists():
        return real
    example = cfg_dir / "servers.example.toml"
    if example.exists():
        log.warning(
            "configs/servers.toml not found; falling back to "
            "configs/servers.example.toml. Copy it to configs/servers.toml "
            "and fill in the placeholders for your local layout."
        )
        return example
    raise FileNotFoundError(
        f"Neither {real} nor {example} exists. Copy "
        "configs/servers.example.toml to configs/servers.toml and fill "
        "in the placeholders for your local layout."
    )

# 100 ms of silence at 16 kHz mono int16. Whisper rejects empty PCM but
# accepts silence; the first call still pays the model-warm-up cost.
_WARMUP_PCM = b"\x00\x00" * 1_600
_WARMUP_TEXT = "hi"

# Stateless validator — fine at module level.
_client_msg = TypeAdapter(ClientMessage)

# Source-checkout layout only: src/glados/core/server.py → repo root is parents[3].
# GLaDOS is self-hosted, not pip-distributed; if that ever changes, ship the
# built client as package data and resolve via importlib.resources.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLIENT_DIST = _REPO_ROOT / "client_web" / "dist"
_CLIENT_INDEX = _CLIENT_DIST / "index.html"
_CLIENT_ASSETS = _CLIENT_DIST / "assets"

_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>GLaDOS</title>"
    "<style>body{font-family:system-ui;background:#0e0f12;color:#e6e9ef;"
    "padding:2rem;max-width:40rem;margin:auto}"
    "code{background:#16181d;padding:.15rem .35rem;border-radius:3px}</style>"
    "<h1>Client not built</h1>"
    "<p>Run <code>cd client_web &amp;&amp; npm install &amp;&amp; npm run build</code>"
    " (or <code>npm run dev</code> for HMR on port 5173).</p>"
)


# ---- Component builders ------------------------------------------------


def _build_llm(cfg: LLMConfig) -> LLM:
    if cfg.backend == "ollama":
        return OllamaLLM(
            host=cfg.host,
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )
    return FakeLLM()


def _build_specialist_llm(
    cfg: RouterConfig, llm_cfg: LLMConfig, primary_llm: LLM
) -> LLM | None:
    """Construct the specialist brain for the v2.6 router, or None when it
    isn't wired. Returns None unless the router is enabled.

    provider="local" is the default all-local path: the specialist runs on a
    local Ollama model. Nothing leaves the box, so it needs neither the cloud
    opt-in nor an API key. An empty `local_smart_model` reuses the already-built
    primary brain instance (identical behaviour — useful purely to see routing/
    escalation fire); point it at a larger tag for a real primary/specialist
    split.

    provider="anthropic" is the dormant cloud escape hatch and fails closed: the
    cloud opt-in must be set AND the API key present in the configured env var
    (TOML never holds the key — ARCH §9). A missing key logs a warning and
    disables the specialist rather than crashing boot. The endpoint is hardcoded
    to api.anthropic.com — reusing this for a self-hosted endpoint is a separate
    guarded slice (ARCH §12)."""
    if not cfg.enabled:
        return None
    if cfg.provider == "local":
        if not cfg.local_smart_model or cfg.local_smart_model == llm_cfg.model:
            return primary_llm  # alias — same instance, same model
        return OllamaLLM(
            host=llm_cfg.host,
            model=cfg.local_smart_model,
            temperature=llm_cfg.temperature,
            timeout=llm_cfg.timeout,
        )
    if not cfg.cloud_enabled:
        return None
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        log.warning(
            "router.cloud_enabled is set but %s is empty — specialist disabled, "
            "all turns run on the primary brain",
            cfg.api_key_env,
        )
        return None
    from ..brain.llm.anthropic import AnthropicLLM

    return AnthropicLLM(api_key=api_key, model=cfg.cloud_model)


def _build_router(cfg: RouterConfig) -> Router | None:
    return Router(max_words_local=cfg.max_words_local) if cfg.enabled else None


def _build_vad(cfg: VADConfig) -> VAD:
    if cfg.backend == "silero":
        from ..audio.vad.silero import SileroVAD

        return SileroVAD(
            threshold=cfg.silero_threshold,
            min_silence_ms=cfg.silero_min_silence_ms,
            speech_pad_ms=cfg.silero_speech_pad_ms,
        )
    return FakeVAD(utterance_samples=cfg.fake_utterance_samples)


def _build_stt(cfg: STTConfig) -> STT:
    if cfg.backend == "faster-whisper":
        from ..audio.stt.whisper import WhisperSTT

        return WhisperSTT(
            model=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            language=cfg.whisper_language,
            hotwords=cfg.whisper_hotwords,
            initial_prompt=cfg.whisper_initial_prompt,
        )
    return FakeSTT(text=cfg.fake_text)


def _build_tts(cfg: TTSConfig) -> TTS:
    if cfg.backend == "piper":
        from ..audio.tts.piper import PiperTTS

        return PiperTTS(
            voice=cfg.piper_voice,
            voices_dir=cfg.piper_voices_dir,
            pronunciations=cfg.pronunciations,
        )
    return FakeTTS()


async def _warmup(stt: STT, tts: TTS) -> None:
    """Hide first-inference latency by exercising each backend once on
    server boot. Real Whisper/Piper load weights at construct time, but
    the *first* `transcribe`/`synthesize` still pays ctranslate2 /
    onnxruntime kernel-graph compilation and thread-pool init (~0.5-1.5 s
    each). Fakes accept the calls cheaply."""
    try:
        await stt.transcribe(_WARMUP_PCM)
    except Exception:
        log.exception("STT warmup failed (continuing — first transcribe may be slow)")
    try:
        async for _ in tts.synthesize(_WARMUP_TEXT):
            pass
    except Exception:
        log.exception("TTS warmup failed (continuing — first synth may be slow)")


# ---- App factory -------------------------------------------------------


def build_app(config_dir: Path | None = None) -> FastAPI:
    """Build a fresh FastAPI app with all components bound to `app.state`.

    `config_dir` defaults to `$GLADOS_CONFIG_DIR` or `"configs"`. Each call
    produces an independent app instance — no shared module-level state —
    so tests can construct isolated apps and the lifespan cleans up
    per-app room workers on shutdown.
    """
    log_path = setup_logging()
    log.info("glados starting (logging to %s)", log_path)
    cfg_dir = config_dir or Path(os.environ.get(CONFIG_DIR_ENV, "configs"))
    glados_cfg = load_glados_config(cfg_dir / "glados.toml")
    rooms_cfg = load_rooms_config(cfg_dir / "rooms.toml")
    servers_cfg = load_servers_config(_resolve_servers_toml(cfg_dir))

    traces = TraceStore(glados_cfg.server.traces_dir)
    mcp = MCPRegistry()
    mcp.register(NowTool())
    for tool in TOY_TOOLS:
        mcp.register(tool)
    # Subprocess MCP servers are wired in the lifespan startup because they
    # need to be spawned + queried inside the running event loop. Tracked
    # here for shutdown:
    stdio_servers: list[StdioServer] = []
    llm = _build_llm(glados_cfg.llm)
    specialist_llm = _build_specialist_llm(glados_cfg.router, glados_cfg.llm, llm)
    router = _build_router(glados_cfg.router)
    # STT and TTS are shared across connections (real backends load model
    # weights on construct). VAD is per-connection — it carries per-stream
    # buffer state — and is built fresh in `_build_pipeline`.
    stt = _build_stt(glados_cfg.stt)
    tts = _build_tts(glados_cfg.tts)
    connections: dict[str, WebSocket] = {}

    async def send(client_id: str, msg: BaseModel) -> None:
        ws = connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(msg.model_dump())
        except Exception:
            # Recipient went away mid-broadcast. Drop the slot so subsequent
            # fan-outs skip it, and never let one dead client take the turn down.
            if connections.get(client_id) is ws:
                del connections[client_id]

    def clients_in_room(room_id: str) -> list[str]:
        return [
            c.client_id
            for c in rooms_cfg.clients
            if c.room_id == room_id and c.client_id in connections
        ]

    sessions = SessionRegistry(idle_window_s=glados_cfg.session.idle_window_s)
    organizer = Organizer(
        llm=llm,
        tts=tts,
        mcp=mcp,
        traces=traces,
        sessions=sessions,
        send=send,
        binding_for_client=rooms_cfg.find,
        clients_in_room=clients_in_room,
        router=router,
        specialist_llm=specialist_llm,
        escalate_on_failed=glados_cfg.router.escalate_on_failed,
        history_max_turns=glados_cfg.session.history_max_turns,
        system_prompt=glados_cfg.llm.system_prompt or None,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Make sure the Ollama daemon is up before the first turn can land.
        # Runs inline (not as a task) because warmup below depends on it
        # being reachable. ensure() is best-effort: a failure here logs and
        # continues, and the user-facing failure mode falls through to the
        # existing httpx ConnectError path in OllamaLLM.chat.
        ollama_lifecycle: OllamaLifecycle | None = _app.state.ollama_lifecycle
        if ollama_lifecycle is not None:
            await ollama_lifecycle.ensure()
        # Background so /healthz answers immediately and a fresh client can
        # connect while warm-up is still finishing. The first audio utterance
        # only benefits if warm-up completed first, but completing in 1-3 s
        # is much better than running it inline at first use.
        #
        # Read stt/tts from `_app.state` at startup rather than closing
        # over the build-time locals — tests routinely swap these via
        # `app.state.stt = fake` after `build_app()` returns and before
        # `with TestClient(app)` fires the lifespan.
        #
        # Only `stt` and `tts` are safely hot-swappable this way.
        # `app.state.organizer` is wired into the build-time `send` /
        # `clients_in_room` closures, so replacing `app.state.organizer`
        # would leave handlers pointing at a new Organizer while the
        # original is still active in its workers. If a test needs a
        # different Organizer, call `build_app()` again with a fresh
        # config.
        task = asyncio.create_task(_warmup(_app.state.stt, _app.state.tts))
        # Spawn autostart stdio MCP servers and register their tools.
        # Done inline (not in a background task) so the registry is
        # populated before the first WS connection can call a tool.
        # Guard-wrapped, hash-approved server memory accumulated across all
        # trusted servers, installed on the organizer once the loop finishes
        # (ARCH §14). Empty unless a server is flagged `trusted` AND ships a
        # `memory://lessons` resource that clears the LocalGuard gate.
        memory_notes: list[str] = []
        # Trusted servers whose lessons failed the hash-approval gate. Metadata
        # only (source/hash/length/reason); surfaced to operators but never
        # injected, never spoken (ARCH §14 BLOCK-notice surface).
        memory_blocks: list[MemoryBlockNotice] = []
        # (server, idle_timeout_s) for every lazy server, handed to the idle
        # reaper after the spawn loop (ARCH §13).
        lazy_servers: list[tuple[StdioServer, float]] = []
        for entry in _app.state.servers_cfg.server:
            if not entry.autostart:
                continue
            # "python" / "python3" resolves against PATH, which on Windows
            # can pick a stranger interpreter that lacks the venv's deps.
            # Substitute the running interpreter so a stdio server sees the
            # same packages GLaDOS itself does.
            command = (
                sys.executable if entry.command in ("python", "python3") else entry.command
            )
            # Run subprocesses from the repo root so relative script paths
            # in servers.toml (e.g. `scripts/toy_stdio_server.py`) resolve
            # regardless of where GLaDOS was launched from.
            server = StdioServer(
                command,
                entry.args,
                env=entry.env,
                cwd=str(_REPO_ROOT),
                server_id=entry.id,
            )
            await server.start()
            try:
                # A wedged third-party server (slow start, broken
                # handshake) would otherwise block GLaDOS boot forever
                # because initialize() has no inherent timeout. 20s is
                # generous for any sane MCP server's handshake and well
                # short of the room worker's patience.
                async with asyncio.timeout(20.0):
                    await server.initialize()
                    specs = await server.list_tools()
            except (Exception, asyncio.TimeoutError) as e:  # noqa: BLE001
                log.warning(
                    "stdio server %s failed to initialise: %s — skipping",
                    entry.id,
                    type(e).__name__ if isinstance(e, asyncio.TimeoutError) else e,
                )
                await server.aclose()
                continue
            for spec in specs:
                overlay = entry.tool_overlays.get(spec.name)
                if overlay is not None:
                    # Real MCP wire shape can't carry GLaDOS-only flags;
                    # apply them from servers.toml here. model_copy keeps
                    # the spec immutable elsewhere. A partial overlay
                    # (only one field in TOML) writes all three from the
                    # ToolOverlay defaults (False / None) — that's fine
                    # because real MCP doesn't define these on the wire
                    # anyway, so the wire side is always at those defaults.
                    spec = spec.model_copy(
                        update={
                            "untrusted": overlay.untrusted,
                            "requires_confirmation": overlay.requires_confirmation,
                            "mutating": overlay.mutating,
                            "timeout_s": overlay.timeout_s,
                        }
                    )
                _app.state.mcp.register(StdioToolProxy(server, spec))
            _app.state.stdio_servers.append(server)
            # Server-shipped lessons (ARCH §14). Only trusted servers are
            # candidates (origin gate); the blob is read over MCP, then must
            # clear the LocalGuard hash-approval gate (content gate) before it
            # is injected. memory_gate.vet returns None for any non-approved
            # outcome, so this fails closed and silently for the common case.
            if entry.trusted:
                blob = await server.read_lessons()
                if blob is not None:
                    # check() shells out to LocalGuard; off-thread so the
                    # subprocess can't stall the warmup task on the loop.
                    result = await asyncio.to_thread(memory_gate.check, entry.id, blob)
                    if result.note is not None:
                        memory_notes.append(result.note)
                    else:
                        # Failed the gate — inject nothing, but make the BLOCK
                        # visible to the operator (metadata only).
                        memory_blocks.append(
                            MemoryBlockNotice(
                                source=result.source,
                                sha256=result.sha256,
                                length=result.length,
                                reason=result.reason or "not approved",
                            )
                        )
            # Lazy servers spawn at boot only to list tools (and read lessons
            # above); put the child dormant now so it isn't held resident until
            # someone actually calls a tool (ARCH §13). The first dispatch wakes
            # it; the reaper below sleeps it again after `idle_timeout_s`.
            if entry.lazy:
                await server.sleep()
                lazy_servers.append((server, entry.idle_timeout_s))
        _app.state.organizer.set_memory_notes(memory_notes)
        _app.state.memory_blocks = memory_blocks
        # One sweeper for all lazy servers — sleeps any that woke for a
        # dispatch and have since gone idle past their timeout.
        reaper_task: asyncio.Task | None = (
            asyncio.create_task(_lazy_reaper(lazy_servers), name="lazy-mcp-reaper")
            if lazy_servers
            else None
        )
        try:
            yield
        finally:
            if reaper_task is not None and not reaper_task.done():
                reaper_task.cancel()
                try:
                    await reaper_task
                except asyncio.CancelledError:
                    pass
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # Stop room workers cleanly. In-flight turns receive CancelledError
            # via their own task and run their finally-block before exit.
            await _app.state.organizer.close()
            # Close LLM HTTP client (OllamaLLM holds a pooled AsyncClient).
            # Fakes don't expose aclose — skip silently. NOTE: llm.aclose()
            # must not require Ollama to be reachable — it only tears down
            # the local httpx pool. We kill the daemon AFTER this step.
            llm_aclose = getattr(_app.state.llm, "aclose", None)
            if llm_aclose is not None:
                await llm_aclose()
            # Specialist brain (v2.6 router) may hold its own pooled
            # AsyncClient. When it aliases the primary llm instance (provider=
            # "local" with no distinct model) it was already closed above — skip
            # it to avoid a double-aclose.
            specialist = _app.state.specialist_llm
            specialist_aclose = getattr(specialist, "aclose", None)
            if specialist_aclose is not None and specialist is not _app.state.llm:
                await specialist_aclose()
            # Tear down stdio MCP subprocesses last so any in-flight tool
            # call gets cancelled by the organizer.close() above before
            # its subprocess vanishes.
            for srv in _app.state.stdio_servers:
                await srv.aclose()
            # Stop Ollama only if GLaDOS started it. A daemon that was
            # already up when we booted may have other consumers.
            if ollama_lifecycle is not None:
                await ollama_lifecycle.stop_if_started()

    app = FastAPI(title="GLaDOS", version="0.1.0", lifespan=lifespan)

    if _CLIENT_ASSETS.is_dir():
        app.mount("/assets", StaticFiles(directory=_CLIENT_ASSETS), name="assets")

    # All runtime state lives on app.state — handlers fetch it via
    # `ws.app.state` / `request.app.state`. No module-level singletons.
    app.state.glados_cfg = glados_cfg
    app.state.rooms_cfg = rooms_cfg
    app.state.servers_cfg = servers_cfg
    app.state.organizer = organizer
    app.state.connections = connections
    app.state.mcp = mcp
    app.state.stdio_servers = stdio_servers
    # Trusted-server memory that failed the LocalGuard gate at load time
    # (ARCH §14). Populated by the lifespan; initialised here so the
    # /admin/memory route and the push-on-connect path are safe before the
    # lifespan has run (e.g. in tests that don't enter the lifespan).
    app.state.memory_blocks = []
    app.state.stt = stt
    app.state.tts = tts
    app.state.llm = llm
    app.state.specialist_llm = specialist_llm
    app.state.router = router
    # Only manage the daemon when we're actually pointed at one. Fakes don't
    # need it; tests run with backend="fake" and get None here, so the
    # lifespan skips both ensure() and stop_if_started().
    app.state.ollama_lifecycle = (
        OllamaLifecycle(glados_cfg.llm.host) if glados_cfg.llm.backend == "ollama" else None
    )
    # VAD has per-stream buffer state, so we hand `_build_pipeline` a
    # factory rather than an instance and build fresh per connection.
    # Stored on state so tests can swap it post-`build_app()` the same
    # way they swap stt/tts.
    app.state.vad_factory = _build_vad
    # Secrets store: defaults to the real OS keyring. Tests inject
    # InMemorySecrets via `app.state.secrets = ...` after build_app().
    app.state.secrets = KeyringSecrets()
    # Handshake admission control (caps + per-IP failure lockout). Tests
    # swap it post-build_app() to drive its injectable clock.
    app.state.handshake_gate = HandshakeGate(glados_cfg.handshake)

    _register_routes(app)
    return app


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


def _require_loopback(request: Request) -> None:
    """Reject non-loopback callers with 403.

    These operator routes leak inventory (the wired tool list, blocked
    server-memory metadata). That was acceptable while the server bound to
    127.0.0.1; once `GLADOS_HOST` exposes it on the LAN, the routes must stay
    loopback-only. The web UI shell (`/`, `/assets`) is intentionally not
    gated — it carries no inventory and clients fetch their data over the
    token-authenticated WS, not these routes."""
    peer = request.client.host if request.client else None
    if peer not in _LOOPBACK_HOSTS:
        raise HTTPException(status_code=403, detail="loopback only")


def _register_routes(app: FastAPI) -> None:
    """Bind HTTP + WS routes to `app`. Each handler reads runtime state
    from `app.state` rather than module globals."""

    @app.get("/")
    async def index() -> Response:
        if _CLIENT_INDEX.is_file():
            return FileResponse(_CLIENT_INDEX)
        return HTMLResponse(_NOT_BUILT_HTML, status_code=503)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict:
        _require_loopback(request)
        s = request.app.state
        return {
            "ok": True,
            "rooms": len({c.room_id for c in s.rooms_cfg.clients}),
            "tools": [spec.qualified for spec in s.mcp.specs()],
        }

    @app.get("/admin/memory")
    async def admin_memory(request: Request) -> dict:
        """Operator view of trusted-server memory that failed the LocalGuard
        gate (ARCH §14). Metadata only — no untrusted blob bytes. Loopback-only
        (see _require_loopback): the metadata is operator inventory, not for
        LAN clients. The review/approve step that would accept a blocked blob
        is a separate, high-friction desktop action, not anything this
        read-only endpoint can grant."""
        _require_loopback(request)
        blocks = request.app.state.memory_blocks
        return {"blocks": [b.model_dump() for b in blocks]}

    @app.websocket("/ws/v1")
    async def ws_v1(ws: WebSocket) -> None:
        await ws.accept()
        state = ws.app.state
        gate: HandshakeGate = state.handshake_gate
        peer_ip = ws.client.host if ws.client else "unknown"
        verdict = gate.admit(peer_ip)
        if verdict is not Verdict.OK:
            await _reject(ws, verdict)
            return
        client_id: str | None = None
        pipeline: AudioPipeline | None = None
        try:
            # The slot taken by admit() above covers exactly the pending
            # (pre-auth) phase: released in the finally below on every exit
            # path — success, auth failure, disconnect, timeout, or
            # cancellation — exactly once, before the serve loop starts.
            try:
                async with asyncio.timeout(state.glados_cfg.handshake.timeout_s):
                    binding = await _handshake(
                        ws, state.glados_cfg, state.rooms_cfg, state.secrets,
                        gate, peer_ip,
                    )
            except TimeoutError:
                log.warning("handshake timed out for %s", peer_ip)
                await _close_quietly(ws)
                return
            finally:
                gate.release(peer_ip)
            if binding is None:
                return
            client_id = binding.client_id
            await _replace_connection(state.connections, client_id, ws)
            # Surface any load-time memory BLOCKs to operator UIs the moment
            # they connect — the notices are emitted at startup, before any
            # client exists, so a connecting `ui` client would otherwise never
            # see them. Metadata only; never to mic/speaker roles (ARCH §14).
            if binding.role == "ui":
                await _push_memory_blocks(ws, state.memory_blocks)
            pipeline = _build_pipeline(
                state.glados_cfg,
                state.stt,
                state.organizer,
                client_id,
                state.vad_factory,
            )
            await _serve(ws, client_id, pipeline, state.organizer)
        except WebSocketDisconnect:
            return
        finally:
            if pipeline is not None:
                await pipeline.close()
            if client_id is not None and state.connections.get(client_id) is ws:
                del state.connections[client_id]
            # In-flight turns continue on their room worker even after this
            # WS goes away — other room members still see Done/Cancelled,
            # and the closure-bound `send` no-ops harmlessly for the now-
            # disconnected client.


# ---- WS helpers (state passed explicitly; no module globals) -----------


def _build_pipeline(
    glados_cfg: GladosConfig,
    stt: STT,
    organizer: Organizer,
    client_id: str,
    vad_factory: Callable[[VADConfig], VAD],
) -> AudioPipeline:
    sink = (
        AudioSink(glados_cfg.server.traces_dir, client_id)
        if glados_cfg.audio.wav_traces
        else None
    )

    async def on_utterance(text: str) -> None:
        await organizer.handle_audio_text(client_id, text)

    return AudioPipeline(
        sink=sink,
        vad=vad_factory(glados_cfg.vad),
        stt=stt,
        on_utterance=on_utterance,
    )


async def _handshake(
    ws: WebSocket,
    glados_cfg: GladosConfig,
    rooms_cfg: RoomsConfig,
    secrets: SecretsStore,
    gate: HandshakeGate,
    peer_ip: str,
):
    raw = await ws.receive_json()
    try:
        msg = _client_msg.validate_python(raw)
    except ValidationError as e:
        await _send_error(ws, "bad_message", str(e))
        await ws.close()
        return None
    if not isinstance(msg, Hello):
        await _send_error(ws, "expected_hello", "first message must be hello")
        await ws.close()
        return None

    # Only the two credential failures below count toward the per-IP
    # lockout. Malformed hellos (above) and binding mismatches (below — the
    # peer holds a valid token, it's just misconfigured) are bounded by the
    # caps and timeout instead, so a fuzzer can't trip the lockout cheaply
    # and a legitimate-but-misconfigured device can't lock itself out.
    if msg.client_id not in glados_cfg.auth.clients:
        gate.record_failure(peer_ip)
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None
    expected = secrets.get("client-tokens", msg.client_id)
    # Constant-time compare so response timing can't leak how many leading token
    # bytes matched — a classic per-byte timing attack that would let an attacker
    # recover the token incrementally instead of searching the full keyspace.
    # Compare as bytes: compare_digest rejects non-ASCII str, and msg.token is
    # attacker-controlled.
    if expected is None or not hmac.compare_digest(expected.encode(), msg.token.encode()):
        gate.record_failure(peer_ip)
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None
    gate.record_success(peer_ip)

    binding = rooms_cfg.find(msg.client_id)
    if binding is None:
        await _send_error(ws, "unbound_client", f"{msg.client_id} not in rooms.toml")
        await ws.close()
        return None
    if binding.room_id != msg.room_id or binding.role != msg.role:
        await _send_error(ws, "binding_mismatch", "room/role differs from rooms.toml")
        await ws.close()
        return None

    return binding


async def _serve(
    ws: WebSocket,
    client_id: str,
    pipeline: AudioPipeline,
    organizer: Organizer,
) -> None:
    while True:
        event = await ws.receive()
        if event["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(event.get("code", 1000))
        if (data := event.get("bytes")) is not None:
            await _handle_audio(ws, pipeline, data)
            continue
        if (text := event.get("text")) is None:
            # ASGI permits a websocket.receive with neither payload set;
            # ignore rather than spin or crash.
            continue
        try:
            msg = _client_msg.validate_json(text)
        except ValidationError as e:
            await _send_error(ws, "bad_message", str(e))
            continue
        if isinstance(msg, UserText):
            # Enqueue on the speaker's room FIFO. Returns immediately so
            # the receive loop stays responsive; the turn runs on the
            # room's worker task.
            await organizer.handle_user_text(client_id, msg.text)
        elif isinstance(msg, Interrupt):
            await organizer.handle_interrupt(client_id, msg.session_id)
        elif isinstance(msg, ToolConfirmResponse):
            await organizer.handle_tool_confirm_response(client_id, msg)


async def _handle_audio(ws: WebSocket, pipeline: AudioPipeline, data: bytes) -> None:
    try:
        await pipeline.feed_frame(data)
    except FrameTooShort as e:
        await _send_error(ws, "bad_audio_frame", str(e))


async def _push_memory_blocks(
    ws: WebSocket, blocks: list[MemoryBlockNotice]
) -> None:
    """Send pending memory-BLOCK notices to a freshly-connected operator UI.
    Best-effort: a send failure here must not abort the handshake, so a dead
    socket just drops the notice (it stays in state for the next connect)."""
    for notice in blocks:
        try:
            await ws.send_json(notice.model_dump())
        except Exception:
            return


async def _reject(ws: WebSocket, verdict: Verdict) -> None:
    """Turn away a connection the gate refused. WS close code 1013
    ("try again later") plus a distinct error code so a locked-out operator
    debugging a misconfigured device sees the truth, not `auth_failed`."""
    code, message = (
        ("server_busy", "too many pending handshakes; retry shortly")
        if verdict is Verdict.BUSY
        else ("rate_limited", "too many failed handshakes; retry later")
    )
    try:
        await _send_error(ws, code, message)
        await ws.close(code=1013)
    except Exception:
        pass  # peer already gone — the reject must not become a 500


async def _close_quietly(ws: WebSocket) -> None:
    try:
        await ws.close(code=1013)
    except Exception:
        pass  # already disconnected or close raced the timeout


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json(ErrorMessage(code=code, message=message).model_dump())


async def _replace_connection(
    connections: dict[str, WebSocket], client_id: str, ws: WebSocket
) -> None:
    prev = connections.get(client_id)
    if prev is not None and prev is not ws:
        try:
            await prev.close()
        except Exception:
            pass
    connections[client_id] = ws


# ---- Module-level default app (back-compat) ----------------------------
#
# Existing callers do `from glados.core.server import app` and hand the
# result to uvicorn or TestClient. Built lazily at import time using the
# default config dir. Tests that want isolation should call build_app()
# directly with a tmp config dir.

app = build_app()
