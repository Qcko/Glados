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
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

from ..audio.pipeline import AudioPipeline
from ..audio.stt.fake import FakeSTT
from ..audio.tts.fake import FakeTTS
from ..audio.vad.fake import FakeVAD
from ..brain.llm.fake import FakeLLM
from ..brain.llm.ollama import OllamaLLM
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
    ServersConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    load_glados_config,
    load_rooms_config,
    load_servers_config,
)
from .organizer import Organizer
from .secrets import KeyringSecrets, SecretsStore
from .protocols import (
    ClientMessage,
    ErrorMessage,
    Hello,
    Interrupt,
    ToolConfirmResponse,
    UserText,
)
from .sessions import SessionRegistry
from .traces import TraceStore


log = logging.getLogger(__name__)

CONFIG_DIR_ENV = "GLADOS_CONFIG_DIR"

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
        )
    return FakeSTT(text=cfg.fake_text)


def _build_tts(cfg: TTSConfig) -> TTS:
    if cfg.backend == "piper":
        from ..audio.tts.piper import PiperTTS

        return PiperTTS(voice=cfg.piper_voice, voices_dir=cfg.piper_voices_dir)
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
    cfg_dir = config_dir or Path(os.environ.get(CONFIG_DIR_ENV, "configs"))
    glados_cfg = load_glados_config(cfg_dir / "glados.toml")
    rooms_cfg = load_rooms_config(cfg_dir / "rooms.toml")
    servers_cfg = load_servers_config(cfg_dir / "servers.toml")

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

    sessions = SessionRegistry()
    organizer = Organizer(
        llm=llm,
        tts=tts,
        mcp=mcp,
        traces=traces,
        sessions=sessions,
        send=send,
        binding_for_client=rooms_cfg.find,
        clients_in_room=clients_in_room,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
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
                await server.initialize()
                specs = await server.list_tools()
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "stdio server %s failed to initialise: %s — skipping",
                    entry.id,
                    e,
                )
                await server.aclose()
                continue
            for spec in specs:
                _app.state.mcp.register(StdioToolProxy(server, spec))
            _app.state.stdio_servers.append(server)
        try:
            yield
        finally:
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
            # Fakes don't expose aclose — skip silently.
            llm_aclose = getattr(_app.state.llm, "aclose", None)
            if llm_aclose is not None:
                await llm_aclose()
            # Tear down stdio MCP subprocesses last so any in-flight tool
            # call gets cancelled by the organizer.close() above before
            # its subprocess vanishes.
            for srv in _app.state.stdio_servers:
                await srv.aclose()

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
    app.state.stt = stt
    app.state.tts = tts
    app.state.llm = llm
    # VAD has per-stream buffer state, so we hand `_build_pipeline` a
    # factory rather than an instance and build fresh per connection.
    # Stored on state so tests can swap it post-`build_app()` the same
    # way they swap stt/tts.
    app.state.vad_factory = _build_vad
    # Secrets store: defaults to the real OS keyring. Tests inject
    # InMemorySecrets via `app.state.secrets = ...` after build_app().
    app.state.secrets = KeyringSecrets()

    _register_routes(app)
    return app


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
        s = request.app.state
        return {
            "ok": True,
            "rooms": len({c.room_id for c in s.rooms_cfg.clients}),
            "tools": [spec.qualified for spec in s.mcp.specs()],
        }

    @app.websocket("/ws/v1")
    async def ws_v1(ws: WebSocket) -> None:
        await ws.accept()
        state = ws.app.state
        client_id: str | None = None
        pipeline: AudioPipeline | None = None
        try:
            binding = await _handshake(
                ws, state.glados_cfg, state.rooms_cfg, state.secrets
            )
            if binding is None:
                return
            client_id = binding.client_id
            await _replace_connection(state.connections, client_id, ws)
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

    if msg.client_id not in glados_cfg.auth.clients:
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None
    expected = secrets.get("client-tokens", msg.client_id)
    if expected is None or expected != msg.token:
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None

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
