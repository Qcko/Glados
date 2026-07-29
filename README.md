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

**v0** — text-only end-to-end. WebSocket protocol, per-client auth, Organizer
with single-turn sessions, JSONL traces, in-process MCP with one dummy tool
(`time.now`), and a real local LLM via **Ollama + Qwen2.5:7b-instruct**. A
deterministic `fake` backend is kept for tests.

Roadmap: v0 text → v1 voice in browser → v2 multi-room concurrency → v3 Pi
clients → v4 wake word → v5 speaker ID. See ARCHITECTURE §12.

## Run

```powershell
uv sync
# one-time: install Ollama and pull the default model
winget install Ollama.Ollama
ollama pull qwen2.5:7b-instruct

uv run glados
```

> **Storage convention.** If you want large caches and model files off the
> system drive, point the relevant env vars at a roomy disk before first
> run. Persist once with `setx` (paths below are examples — substitute your own):
>
> ```powershell
> setx OLLAMA_MODELS         "<data-drive>\ollama\models"
> setx UV_PYTHON_INSTALL_DIR "<data-drive>\uv\python"
> setx UV_CACHE_DIR          "<data-drive>\uv\cache"
> setx HF_HOME               "<data-drive>\hf"
> ```
>
> If Ollama was already pulled to its default path, stop the service, move
> `~/.ollama/models` to your new `OLLAMA_MODELS` location, then start it again.

<!-- rule-guard:allow no-local-env-leak - the "/home/<user>/" below is an
     anonymised placeholder in prose describing this very check, not a path
     from anyone's machine. -->

> **Git hooks (one-time).** The repo ships a `pre-commit` guard in
> [.githooks/](.githooks/) that blocks committing machine-local paths (drive
> letters, `/home/<user>/`, …) into tracked files. Point git at it once per
> clone:
>
> ```powershell
> git config core.hooksPath .githooks
> ```
>
> It scans only newly-added lines, skips `*.example.*`/`*.template.*`, and a
> line tagged `allow-local-path` is exempt. Set `GLADOS_PATHGUARD_MODE=warn`
> to print instead of block.

Build the web client once before opening the UI:

```powershell
cd client_web
npm install
npm run build
```

Then open http://127.0.0.1:8765/ for the UI, or use `ws://127.0.0.1:8765/ws/v1`
directly. `GET /healthz` for a liveness check. Tests: `uv run pytest`
(integration tests skip when Ollama isn't reachable).

### Admin room-viewer (loopback-only debugging)

A separate, **loopback-only** surface lets you watch any room's conversation as
text — useful for debugging a remote room client (e.g. a phone). It is off by
default and, by design, never reachable from the LAN (ARCHITECTURE §9). To
enable it, stage a distinct admin secret in the keyring and set its port:

```powershell
python -m glados.secrets set admin observe-token   # paste a strong secret when prompted
setx GLADOS_ADMIN_PORT 9765                         # or set for the session before `uv run glados`
```

Then open **http://127.0.0.1:9765/** on the server box, enter that secret, and
pick a room. The view is read-only (it cannot inject into a room), text-only (no
audio), and strips tool arguments/results. Leave `GLADOS_ADMIN_PORT` unset to
keep the surface disabled.

For client iteration, run Vite's dev server with HMR — it proxies the WS to
the FastAPI backend running on 8765/8000:

```powershell
cd client_web; npm run dev   # then open http://127.0.0.1:5173
```

To run with the deterministic fake LLM (no Ollama needed):

```powershell
$env:GLADOS_LLM_BACKEND = "fake"; uv run glados
```

`GLADOS_LLM_MODEL` and `GLADOS_LLM_HOST` also work as overrides.

To retune persona/verbosity without editing code, set `[llm].system_prompt`
in `configs/glados.toml` (empty uses the built-in default). The
untrusted-content rule is force-appended to any override, so it can't be
dropped by accident.

> uv stores its Python and cache on the same drive as the repo (cross-drive
> renames fail on Windows). If the repo lives off the system drive, point
> `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` at the same drive first.

Configs live in [configs/](configs/): `glados.toml` for auth tokens,
`rooms.toml` for client → room/role bindings, `servers.toml` for the
MCP subprocess servers GLaDOS spawns.

`servers.toml` is **gitignored** because it carries machine-specific
paths. Before first run, copy the tracked template:

```powershell
cp configs/servers.example.toml configs/servers.toml
# then edit configs/servers.toml — replace every <path-to> / <your-secrets-dir>
```

If `servers.toml` is missing, GLaDOS falls back to the example with a
warning so smoke-tests still work; for real use, copy and customize.
