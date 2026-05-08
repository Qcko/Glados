# GLaDOS

Local-first, voice-controlled AI assistant for the home. Runs on your own
hardware, talks to your own services, and never sends conversations to a
third-party cloud.

## What it does

A wake word, a question or a command, an answer back through the speakers.
Under the hood, a local LLM picks the right tool from a set of MCP servers
and runs it:

- **LifeQuests** — personal quest / habit app
- **Dunnes Stores** — grocery ordering via the website
- **Spotify** — playback control
- **Web search** — privacy-preserving meta-search (SearXNG)
- **Calendar** — local CalDAV
- **Cooking app** — meal planning (planned)

Adding a new capability means plugging in another MCP server. Adding a new
microphone or speaker means pairing another thin client. The brain is one
codebase.

## How it's built

- **Server** — orchestrator, Organizer, LLM, MCP manager, memory, STT, TTS.
  One beefy box.
- **Clients** — dumb transducers (one mic, one speaker, or one UI). Browser
  today, Raspberry Pis in each room later.
- **Organizer** — the only thing that thinks about rooms, sessions, dedup,
  and routing. Two people in two rooms talking at once is the default case.
- **Pluggable** — small adapter Protocols for STT / TTS / LLM / MCP, swapped
  via config.

Full design and trade-offs in [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

**v0** — text-only client/server loop. WebSocket protocol, per-client auth,
config loader, and an echo round-trip are in. LLM and first MCP server land
next.

Roadmap: v0 text → v1 voice in browser → v2 multi-room concurrency → v3 Pi
clients → v4 wake word → v5 speaker ID. See ARCHITECTURE §12.

## Run

```powershell
uv sync
uv run glados
```

Server on `ws://127.0.0.1:8765/ws/v1`. `GET /healthz` for a liveness check.
Tests: `uv run pytest`.

> uv stores its Python and cache on the same drive as the repo (cross-drive
> renames fail on Windows). If the repo lives on E:, set
> `UV_PYTHON_INSTALL_DIR=E:/uv/python` and `UV_CACHE_DIR=E:/uv/cache` first.

Configs live in [configs/](configs/): `glados.toml` for auth tokens,
`rooms.toml` for client → room/role bindings.
