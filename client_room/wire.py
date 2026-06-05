"""Vendored slice of the GLaDOS `/ws/v1` wire contract.

This is a deliberate, tested duplication of the constants and frame layout
defined server-side in `src/glados/core/protocols.py`. The room client must be
installable and runnable on a device that has no copy of the server package, so
it cannot import `glados.*`. `tests/test_client_room.py` carries a drift-guard
that asserts these values still match the server's, so the duplication can't
silently rot.

Audio frame layout (binary WebSocket frame):
    bytes 0..4   : big-endian uint32 sequence number
    bytes 4..end : PCM16-LE samples at AUDIO_SAMPLE_RATE Hz, mono
"""

from __future__ import annotations

import struct

WS_PATH = "/ws/v1"

# Must match src/glados/core/protocols.py (guarded by a drift test).
AUDIO_SAMPLE_RATE = 16_000
AUDIO_HEADER_LEN = 4

# 50 ms of audio at 16 kHz. Matches client_web/src/audio/processor.js so the
# server's VAD sees the same chunking the browser produces.
BATCH_SAMPLES = 800

# uint32 wrap mask — mirrors the browser's `(seq + 1) >>> 0`.
_SEQ_MASK = 0xFFFFFFFF


def hello(client_id: str, room_id: str, role: str, token: str) -> dict:
    """Build the handshake message. Sent as the FIRST message over the socket,
    as JSON text, before any binary audio (the server's `_handshake` calls
    `receive_json()` and rejects a binary first frame). A successful handshake
    is answered with silence — `welcome` is a per-turn broadcast, not a connect
    ack — so the client must NOT wait for a reply before streaming."""
    return {
        "type": "hello",
        "client_id": client_id,
        "room_id": room_id,
        "role": role,
        "token": token,
    }


def frame(seq: int, pcm: bytes) -> bytes:
    """Prefix a PCM16-LE payload with its big-endian uint32 sequence number.
    `seq` is masked to 32 bits so a long-running client wraps cleanly instead
    of overflowing the struct pack."""
    return struct.pack(">I", seq & _SEQ_MASK) + pcm
