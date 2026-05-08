# GLaDOS

Local-first voice assistant. Pluggable STT / TTS / LLM / MCP. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Status: v0 (text echo, handshake, protocol)

The WebSocket protocol, config loader, and per-client auth are wired up. No
LLM or MCP yet — `user_text` round-trips as `echo: <text>`.

## Setup

uv-managed project. uv stores its Python and cache on the same drive as the
repo (cross-drive renames fail on Windows), so set:

```powershell
$env:UV_PYTHON_INSTALL_DIR = "E:/uv/python"
$env:UV_CACHE_DIR = "E:/uv/cache"
```

Install deps:

```powershell
uv sync
```

## Run

```powershell
uv run glados
```

Server listens on `ws://127.0.0.1:8765/ws/v1`. Health: `GET /healthz`.

## Test

```powershell
uv run pytest
```

## Talking to the server

Send `hello` first, then `user_text`:

```json
{"type":"hello","client_id":"desk-ui","room_id":"desk","role":"ui","token":"dev-token-desk"}
{"type":"user_text","text":"hello"}
```

Tokens and client → room/role bindings live in [configs/glados.toml](configs/glados.toml)
and [configs/rooms.toml](configs/rooms.toml). v0 stores tokens in plaintext
config; OS-keyring storage lands in v1+.
