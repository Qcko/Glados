# TODO

State of the project at a glance. Source of truth lives in code + commits;
this file is for "where do we pick up next session?".

## Done (v0)

- **Step 1** — Pydantic wire protocol, config loader, FastAPI `/ws/v1`,
  handshake with per-client tokens + room/role binding.
- **Step 2** — Organizer (single-turn sessions keyed by `(room_id, speaker_id)`),
  SessionRegistry hook, MCP registry with timeout + per-session envelope,
  fake LLM, JSONL traces.
- **Step 2.5** — Tiny HTML demo client served at `GET /`.
- **Step 3** — Ollama adapter (Qwen2.5:7b-instruct) behind the same `LLM`
  Protocol; config-driven backend swap (`fake` vs `ollama`); env overrides.
- **Storage** — E: drive convention adopted globally; Ollama models moved.

32 tests green (30 unit + 2 integration); live demo verified via smoke
script and HTML client.

## Pick up next (choose one)

- **v0 step 4 — first real MCP server.** Wire LifeQuests as a real MCP via
  stdio transport. Proves the multi-server pluggability claim and is small.
  Likely needs: `mcp/stdio_client.py`, server-process lifecycle, env/secret
  plumbing. Smallest viable path.
- **v1 — voice in the browser.** Bigger. Browser `MediaRecorder` → Opus/WebM
  → server-side decode → faster-whisper → existing Organizer → Piper TTS →
  browser playback. Includes the `interrupt`/`cancelled` wiring and AEC
  questions for v2 Pi clients.

Recommendation: do step 4 first — it's a session-sized chunk and locks in
that the Organizer/MCP boundary holds for a non-trivial server before voice
adds its own complexity.

## Parked (revisit when relevant)

- **`<tool_result>...</tool_result>` delimiters** on tool content sent to
  the LLM. Mandatory before any web-tool MCP lands (ARCHITECTURE §7
  untrusted-content). TODO comment is in `brain/llm/ollama.py`.
- **Connection pooling on OllamaLLM** — hoist `httpx.AsyncClient` to
  instance scope with explicit `aclose()`. Trade-off: barge-in latency vs.
  v0 simplicity.
- **`build_app(config_dir)` factory** — replaces module-level singletons in
  `core/server.py`. Pays off when a second app instance is needed.
- **TraceWriter "a" mode** — needed once SessionRegistry holds multi-turn
  sessions (currently truncates on re-open).
- **Move `_SYSTEM_PROMPT`** out of `core/organizer.py` into
  `brain/prompts/` once that package exists.
- **OS keyring for secrets** (Windows Credential Manager) — replaces the
  plaintext `[auth.tokens]` table in `glados.toml`. Lands when the first
  real third-party credential (Spotify OAuth, Dunnes login) arrives.
- **Per-client pairing flow** with one-time short codes — replaces dev
  tokens in config when the first non-localhost client appears.

## Known unknowns

See [ARCHITECTURE.md §13](ARCHITECTURE.md#13-known-unknowns) — the strategic
list (skills layer, tool-calling benchmark vs vLLM, Pi update channel,
voice-cloning legality, vector store earn-its-keep test, backup story,
family onboarding UX). Not session-sized; revisit before relevant version.
