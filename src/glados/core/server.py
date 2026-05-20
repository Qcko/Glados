"""FastAPI server exposing `/ws/v1`.

Handshake → register client connection → forward `user_text` to the
Organizer. The Organizer owns sessions, tool dispatch, and egress; this file
only handles wire I/O and connection bookkeeping.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
from ..servers.time_server import NowTool
from ..servers.toy_server import TOY_TOOLS
from .adapters import LLM, STT, TTS, VAD
from .audio_sink import AudioSink, FrameTooShort
from .config import (
    GladosConfig,
    LLMConfig,
    RoomsConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    load_glados_config,
    load_rooms_config,
)
from .organizer import Organizer
from .protocols import (
    ClientMessage,
    ErrorMessage,
    Hello,
    Interrupt,
    UserText,
)
from .sessions import SessionRegistry
from .traces import TraceStore


log = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))

# 100 ms of silence at 16 kHz mono int16. Whisper rejects empty PCM but
# accepts silence; the first call still pays the model-warm-up cost.
_WARMUP_PCM = b"\x00\x00" * 1_600
_WARMUP_TEXT = "hi"

_glados_cfg: GladosConfig = load_glados_config(CONFIG_DIR / "glados.toml")
_rooms_cfg: RoomsConfig = load_rooms_config(CONFIG_DIR / "rooms.toml")
_client_msg = TypeAdapter(ClientMessage)

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


_traces = TraceStore(_glados_cfg.server.traces_dir)
_mcp = MCPRegistry()
_mcp.register(NowTool())
for _tool in TOY_TOOLS:
    _mcp.register(_tool)
_llm: LLM = _build_llm(_glados_cfg.llm)
# STT and TTS shared across connections (real backends load a model on
# construct). VAD is per-connection — it carries per-stream buffer state.
_stt: STT = _build_stt(_glados_cfg.stt)
_tts: TTS = _build_tts(_glados_cfg.tts)
_connections: dict[str, WebSocket] = {}


async def _send(client_id: str, msg: BaseModel) -> None:
    ws = _connections.get(client_id)
    if ws is None:
        return
    try:
        await ws.send_json(msg.model_dump())
    except Exception:
        # Recipient went away mid-broadcast. Drop the slot so subsequent
        # fan-outs skip it, and never let one dead client take the turn down.
        if _connections.get(client_id) is ws:
            del _connections[client_id]


def _clients_in_room(room_id: str) -> list[str]:
    return [
        c.client_id
        for c in _rooms_cfg.clients
        if c.room_id == room_id and c.client_id in _connections
    ]


_sessions = SessionRegistry()
_organizer = Organizer(
    llm=_llm,
    tts=_tts,
    mcp=_mcp,
    traces=_traces,
    sessions=_sessions,
    send=_send,
    binding_for_client=_rooms_cfg.find,
    clients_in_room=_clients_in_room,
)

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Background so /healthz answers immediately and a fresh client can
    # connect while warm-up is still finishing. The first audio utterance
    # only benefits if warm-up completed first, but completing in 1-3 s
    # is much better than running it inline at first use.
    task = asyncio.create_task(_warmup(_stt, _tts))
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
        await _organizer.close()


app = FastAPI(title="GLaDOS", version="0.1.0", lifespan=_lifespan)
# Source-checkout layout only: src/glados/core/server.py → repo root is parents[3].
# GLaDOS is self-hosted, not pip-distributed; if that ever changes, ship the
# built client as package data and resolve via importlib.resources.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLIENT_DIST = _REPO_ROOT / "client_web" / "dist"
_CLIENT_INDEX = _CLIENT_DIST / "index.html"
_CLIENT_ASSETS = _CLIENT_DIST / "assets"

if _CLIENT_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_CLIENT_ASSETS), name="assets")


_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>GLaDOS</title>"
    "<style>body{font-family:system-ui;background:#0e0f12;color:#e6e9ef;"
    "padding:2rem;max-width:40rem;margin:auto}"
    "code{background:#16181d;padding:.15rem .35rem;border-radius:3px}</style>"
    "<h1>Client not built</h1>"
    "<p>Run <code>cd client_web &amp;&amp; npm install &amp;&amp; npm run build</code>"
    " (or <code>npm run dev</code> for HMR on port 5173).</p>"
)


@app.get("/")
async def index() -> Response:
    if _CLIENT_INDEX.is_file():
        return FileResponse(_CLIENT_INDEX)
    return HTMLResponse(_NOT_BUILT_HTML, status_code=503)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "rooms": len({c.room_id for c in _rooms_cfg.clients}),
        "tools": [s.qualified for s in _mcp.specs()],
    }


@app.websocket("/ws/v1")
async def ws_v1(ws: WebSocket) -> None:
    await ws.accept()
    client_id: str | None = None
    pipeline: AudioPipeline | None = None
    try:
        binding = await _handshake(ws)
        if binding is None:
            return
        client_id = binding.client_id
        await _replace_connection(client_id, ws)
        pipeline = _build_pipeline(client_id)
        await _serve(ws, client_id, pipeline)
    except WebSocketDisconnect:
        return
    finally:
        if pipeline is not None:
            await pipeline.close()
        if client_id is not None and _connections.get(client_id) is ws:
            del _connections[client_id]
        # In-flight turns continue on their room worker even after this
        # WS goes away — other room members still see Done/Cancelled,
        # and `_send` no-ops harmlessly for the now-disconnected client.


def _build_pipeline(client_id: str) -> AudioPipeline:
    sink = (
        AudioSink(_glados_cfg.server.traces_dir, client_id)
        if _glados_cfg.audio.wav_traces
        else None
    )

    async def on_utterance(text: str) -> None:
        await _organizer.handle_audio_text(client_id, text)

    return AudioPipeline(
        sink=sink,
        vad=_build_vad(_glados_cfg.vad),
        stt=_stt,
        on_utterance=on_utterance,
    )


async def _handshake(ws: WebSocket):
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

    expected = _glados_cfg.auth.tokens.get(msg.client_id)
    if expected is None or expected != msg.token:
        await _send_error(ws, "auth_failed", "unknown client or bad token")
        await ws.close()
        return None

    binding = _rooms_cfg.find(msg.client_id)
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
            await _organizer.handle_user_text(client_id, msg.text)
        elif isinstance(msg, Interrupt):
            await _organizer.handle_interrupt(client_id, msg.session_id)


async def _handle_audio(ws: WebSocket, pipeline: AudioPipeline, data: bytes) -> None:
    try:
        await pipeline.feed_frame(data)
    except FrameTooShort as e:
        await _send_error(ws, "bad_audio_frame", str(e))


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json(ErrorMessage(code=code, message=message).model_dump())


async def _replace_connection(client_id: str, ws: WebSocket) -> None:
    prev = _connections.get(client_id)
    if prev is not None and prev is not ws:
        try:
            await prev.close()
        except Exception:
            pass
    _connections[client_id] = ws
