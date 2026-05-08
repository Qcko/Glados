# GLaDOS — Architecture

Local-first voice assistant. Pluggable STT / TTS / LLM / MCP servers. No data
leaves the machine except where a specific integration intrinsically requires
it (Spotify API, Dunnes web, web search query).

---

## 1. Top-level shape — Orchestrator + adapter protocols

A single Python core defines small interfaces (`STT`, `TTS`, `LLM`, `WakeWord`,
`MemoryStore`). Implementations live behind those interfaces and are selected
by config. MCP servers stay as child processes — that is already MCP's native
model.

Alternatives considered:

- **Monolithic app** — simplest, lowest latency, but one crash kills the voice
  loop and component swaps mean code edits.
- **Message bus (NATS / Redis / ZeroMQ)** — maximum pluggability, polyglot,
  but real complexity for a single-user local app: extra hops, schema
  versioning, ops burden. Overkill.
- **Orchestrator + adapters (chosen).** Pluggable where it matters; MCP
  transport already gives process isolation for the riskiest bits (Dunnes
  scraper, Spotify); easy to test with fakes. Trade-off: core is Python-only —
  acceptable, every relevant ML lib is Python-first.

---

## 2. Client / server split

GLaDOS is a **server** (one beefy box) plus **many dumb clients**. A client is
a single transducer — one mic, one speaker, or one UI — not "a Pi" or "a
room". A Pi with a mic array and speakers runs several client processes.

- **Server:** orchestrator, **Organizer** (§3), LLM, MCP manager + servers,
  memory, STT, TTS.
- **Client:** one of `role ∈ {mic, speaker, ui}`. Mic clients stream
  `audio_chunk` up. Speaker clients receive `tts_chunk` down. UI clients send
  `user_text` and render `assistant_delta`. Each client declares
  `(client_id, room_id, role)` at pairing time; that is *all* a client knows.
- **Contract:** versioned WebSocket protocol (`/ws/v1`). Message types:
  `user_text`, `audio_chunk`, `assistant_delta`, `tts_chunk`, `tool_call`,
  `tool_result`, `done`, `error`, `interrupt`, `cancelled`.
- **Auth:** per-client tokens issued via a one-time pairing flow (short code
  on server console). TLS on WebSocket even on LAN. No anonymous clients, no
  shared bearer.
- **Why dumb clients:** the room/session/dedup logic lives in one place
  (Organizer), so adding a hardware button, a phone PWA, or a second mic in a
  room is "pair another client", not a code change.

---

## 3. Organizer & sessions

The **Organizer** sits between dumb clients and the orchestrator. It is the
*only* place rooms, sessions, dedup, queueing, and routing are reasoned
about. Two simultaneous commands from different rooms is the default case,
not an edge case.

### Responsibilities

1. **Ingress.** Tag every incoming `audio_chunk` / `user_text` with
   `(client_id, room_id, role, server_timestamp)`. Server-side timestamps to
   sidestep client clock skew.
2. **Dedup.** Sliding window (~300 ms) across all mic clients. Energy + rough
   audio fingerprint match → keep the loudest, drop the rest. Same code path
   whether duplicates are in one room (kitchen + open-plan living room mics)
   or accidental across the house. UI `user_text` is never deduped.
3. **Session assignment.** Bind the surviving utterance to a
   `(room_id, speaker_id)` session. `speaker_id` defaults to
   `room.default_user` until voice-ID lands. Attach to an existing live
   session if one is open in that room within the conversation idle window,
   else open a new one.
4. **Queue & schedule.** Per-session FIFO so one room's commands stay
   ordered. Cross-session: parallel if the LLM backend supports it (vLLM,
   llama.cpp parallel slots); fair round-robin if not. Short utterances
   matching `stop|cancel|halt` jump the queue and turn into `interrupt` for
   the active session in the same room.
5. **Egress routing.** TTS for session X fans out to every speaker client
   tagged with `room_id == session.room_id`. Speaker selection is a policy
   ("nearest", "all in room", "follow user") configurable per room — not
   baked into clients.
6. **Barge-in.** `interrupt` from any mic in a room cancels that room's
   active session only. Other rooms unaffected.

### Session rules

- **Keyed by `(room_id, speaker_id)`.** Never by `client_id`.
- **Per-session envelope on every MCP call.** `(room_id, speaker_id)` rides
  with each tool invocation so `spotify.play` targets the right room's
  device and `calendar.create` lands on the right user's calendar.
  Protocol-level field, not per-server convention.
- **Permission gates are per-user.** A confirm from one speaker does not
  authorise another. Confirms are spoken back to the room that initiated.
- **No context bleed.** Memory, prompts, and tool results are isolated per
  session.

### Concurrency consequences worth flagging

- **LLM is the bottleneck.** One GPU, two simultaneous tool-calling requests.
  Ollama serialises; vLLM and llama.cpp (parallel slots) batch. With
  Organizer + multi-room as the model, vLLM is justified by concurrency, not
  just tool-calling quality.
- **Barge-in needs client-side AEC.** Mic stays hot during TTS; without AEC
  (PipeWire + `webrtc-audio-processing`) the speaker→mic loop self-triggers
  `interrupt`.
- **Organizer is now load-bearing.** Bugs here break everything. Needs solid
  tests and JSONL traces from v0.

---

## 4. Pipeline

```
[client]                                [server]
 mic → audio_chunk ──────────────►  VAD → STT → ┐
                                                ├─→ LLM (+MCP tool schemas) ⇄ MCP Manager ⇄ {LifeQuests, Dunnes, Spotify, ...}
                            memory (short+long) ┘                              │
 speakers ◄──── tts_chunk ◄──── TTS ◄──── assistant_delta ◄────────────────────┘
              ▲ interrupt → cancelled (drain TTS, kill LLM job)
```

Stream wherever possible: STT partials feed LLM prefill; LLM tokens feed TTS
chunks. Latency SLA: 2 s wake-to-first-audio; 1.2 s stretch goal.

---

## 5. Component choices (each swappable)

| Layer       | Default                                                      | Alternatives                       | Trade-off |
|-------------|--------------------------------------------------------------|------------------------------------|-----------|
| Wake word   | **openWakeWord** (custom "Hey GLaDOS")                       | Porcupine, push-to-talk            | openWakeWord is fully OSS + trainable; Porcupine more accurate but licensed |
| VAD         | **silero-vad**                                               | webrtcvad                          | Silero handles noise better, ~5 ms cost |
| STT         | **faster-whisper** (large-v3 GPU, distil-medium CPU)         | whisper.cpp, Parakeet              | faster-whisper = best Python ergonomics; whisper.cpp wins for embedded targets |
| LLM         | **vLLM** + Qwen2.5-32B-Instruct or Llama 3.3 70B Q4          | llama.cpp parallel slots, Ollama   | vLLM = real concurrency for multi-room and stronger tool-calling than Ollama; benchmark grammar-constrained tool use before locking in |
| TTS         | **Piper** (low latency) → **Kokoro/XTTS** for cloned voice   | Coqui XTTS-v2                      | Piper streams <100 ms; XTTS better quality, needs GPU |
| AEC         | **webrtc-audio-processing** on client (PipeWire)             | speex-aec                          | Required for barge-in; client-side avoids round-trip |
| Memory      | **SQLite** (sessions + FTS5) + optional **LanceDB** later    | Chroma, Qdrant local               | Start with FTS5; add vector store only when retrieval recall demands it |
| Web search  | **SearXNG** self-hosted (Docker), wrapped as MCP             | Brave API                          | SearXNG: no API key, aggregates public engines |
| Secrets     | **OS keyring** (Windows Credential Manager / libsecret)      | encrypted file                     | TOML stores handles only, never values |
| Config      | TOML + env overrides                                         | YAML                               | TOML has cleaner schema |

---

## 6. Adapter protocols (the contract)

Keep them tiny — that is what makes swapping painless.

```python
class STT(Protocol):
    async def transcribe(self, pcm: AsyncIterator[bytes]) -> AsyncIterator[Partial]: ...

class TTS(Protocol):
    async def speak(self, text: AsyncIterator[str]) -> AsyncIterator[bytes]: ...  # PCM

class LLM(Protocol):
    async def chat(self, msgs: list[Msg], tools: list[ToolSpec]) -> AsyncIterator[Delta]: ...

class WakeWord(Protocol):
    async def listen(self) -> AsyncIterator[WakeEvent]: ...
```

Backends that cannot stream are wrapped to yield a single chunk so the
interface stays uniform. All adapter calls accept a cancellation token so
`interrupt` propagates end-to-end.

---

## 7. MCP Manager

Owns lifecycle of MCP servers; exposes their tools as a flat registry to the
LLM.

- **Transport:** stdio for local servers (LifeQuests, cooking, calendar,
  SearXNG wrapper). HTTP/SSE only when a server already runs as a daemon.
- **Config-driven:** `servers.toml` lists each server's command, args, env,
  secret-handles, autostart, restart policy.
- **Tool namespacing:** every tool prefixed with server id
  (`lifequests.create_quest`, `dunnes.add_to_basket`) so overlapping names
  cannot collide.
- **Per-turn frozen registry.** A new MCP server starting mid-turn does not
  mutate the LLM's tool list inside that turn — small models degrade hard
  when the system prompt churns. New tools become visible on the next turn,
  with a one-line "tools updated" note injected.
- **Registry size cap.** Above ~30 tools small models start hallucinating
  tool names; either prune by session relevance or upgrade the model.
- **Per-tool timeouts and circuit breaker.** Default 8 s timeout, cancellable;
  N consecutive failures opens the breaker and the tool is reported to the
  LLM as unavailable so it can apologise coherently. Prevents a Dunnes
  captcha hang from blocking the voice loop.
- **Permission gates on every side-effecting tool,** not only money/messages.
  Quest creation, calendar writes, basket changes, search refinement that
  triggers further tools — all gated. Hard-coded list, not LLM-decided.
  Confirms are per-user (§3).
- **Untrusted-content discipline.** Output of search and any scraped page is
  wrapped in `<external>` delimiters before reaching the LLM, with a system
  rule that instructions inside `<external>` are data, not commands.
  Consider a separate "reader" LLM call (no tools) to summarise external
  content before it touches the tool-armed planner.
- **Health & isolation:** each server is its own subprocess. Crash → manager
  restarts; voice loop keeps running; affected tools temporarily unavailable.

---

## 8. Memory & context window

- **Hot ring buffer in RAM** — last N turns per session, fast path.
- **SQLite session log** — durable; FTS5 covers most retrieval.
- **Rolling summarisation triggered by token budget,** not nightly. When a
  session approaches the model's usable context (typical Q4 32B: 8–32k),
  fold the oldest turns into a per-session "facts" doc that the
  summariser maintains.
- **Validate before v2** with a synthetic 200-turn conversation; if recall
  suffers, add LanceDB + bge-small.

---

## 9. Privacy & security boundaries

| Component                   | Stays local | Crosses network         | Notes |
|-----------------------------|-------------|-------------------------|-------|
| STT / TTS / LLM / wake / VAD| ✅          | —                       | |
| LifeQuests / cooking        | ✅          | —                       | |
| Calendar                    | depends     | only if Google/Outlook  | CalDAV (Radicale) keeps it local |
| Spotify                     | —           | ✅ (unavoidable)        | Only commands + OAuth token; no transcripts |
| Dunnes ordering             | —           | ✅ (unavoidable)        | Scraper runs locally; only HTTP to Dunnes |
| Web search                  | partial     | ✅ (query only)         | SearXNG hides identity; query itself unavoidable |

Invariants:

- *Transcripts, memory, and reasoning never appear in any outbound HTTP body.*
  MCP servers receive only the structured arguments the LLM produces.
- *External content is data, never instructions.* See §7 untrusted-content
  discipline; this is the main mitigation against indirect prompt injection.
- *Secrets live in the OS keyring.* Config holds handles. No plaintext
  credentials in `glados.toml` or environment dumps.
- *Auth is per-client, revocable.* Token loss compromises one client, not the
  house.

---

## 10. Observability

- **One JSONL trace per turn**: wake → STT → LLM → tool calls → TTS spans,
  with timings and tool I/O. Local file, no remote sink.
- **Replay tab** in the web client to step through a turn after the fact.
- **Redaction flag** for sharing traces externally (strips transcripts and
  tool arguments).

Without this we cannot debug "why did it call the wrong tool".

---

## 11. Repo layout

```
glados/
  core/
    orchestrator/   per-session pipeline driver
    organizer/      ingress, dedup, sessions, queue, egress routing
    protocols.py    adapter interfaces
    config.py
    sessions.py
  audio/          wake/, vad/, stt/, tts/   (one module per backend)
  brain/          llm/, memory/, prompts/
  mcp/            manager, registry, permission gates, timeouts
  servers/        lifequests/, dunnes/, spotify/, search/, calendar/, cooking/
  client_web/     browser client (text v0, audio v1)
  client_pi/      Pi mic / speaker client (v3)
  configs/        glados.toml, servers.toml, rooms.toml, voices/
  traces/         per-turn JSONL
  tests/          fakes for every adapter, organizer scenarios, end-to-end
```

---

## 12. Roadmap (small slices, demoable each)

Each version is its own session-sized chunk. Heavy work is deferred.

- **v0 — text-only client/server loop with a real Organizer.** No longer a
  trivial spike: Organizer is load-bearing, so it gets built (in skeleton
  form) on day one even when there is only one client.
  1. Protocols + config loader; `rooms.toml` mapping clients → rooms/roles.
  2. FastAPI server with `/ws/v1`. Schema: `user_text`, `assistant_delta`,
     `tool_call`, `tool_result`, `done`, `error`, `interrupt`, `cancelled`.
  3. Per-client token pairing flow.
  4. **Organizer skeleton:** ingress tagging, session assignment by
     `(room_id, speaker_id)`, per-session FIFO queue, egress routing by
     `room_id`. Dedup and audio fingerprinting stubbed (passthrough); the
     interfaces exist so v1/v3 plug in without refactor.
  5. Minimal web client tagged as `role=ui, room_id=desk`: text box,
     streaming response, quick-action buttons.
  6. LLM adapter (Ollama for v0; swap to vLLM in v2 if concurrency demands).
  7. MCP manager with one dummy server exposing `get_time`. Per-tool timeout.
     Per-session envelope on the call.
  8. JSONL trace per turn — the only way Organizer bugs are debuggable.
  9. End-to-end demo: type / click → streamed answer that called a tool.
  10. Open a second browser tab as `room_id=desk2` to verify session
      isolation, even before voice.
  11. Add LifeQuests MCP as the pluggability test.

- **v1 — voice in the browser.** Browser captures audio. Server-side decode
  of Opus/WebM (browsers do not give raw PCM) into faster-whisper. TTS audio
  streams back. `interrupt` wired up. Validate the protocol against real
  codec realities here, before the Pi client.

- **v2 — concurrency + second MCP.** Switch LLM to vLLM if needed. Add
  Spotify MCP. Two browser tabs as "rooms"; verify isolation.

- **v3 — Pi room client.** Python + PortAudio + WebSocket + AEC. systemd
  unit, auto-update from server. One Pi per room.

- **v4 — wake word + dedup.** openWakeWord on Pi clients; same-utterance
  arbitration server-side.

- **v5 — speaker ID.** Resemblyzer or pyannote to populate `speaker_id`.

---

## 13. Known unknowns (parked, revisit before relevant version)

- **Tool-calling reliability:** benchmark Ollama vs. vLLM/llama.cpp grammar
  before v2.
- **Skills layer:** still skipped. Reconsider when a multi-step routine
  (morning routine, weekly shop) repeatedly fumbles.
- **Pi update channel:** systemd + `/client/latest` endpoint sketched, not
  designed.
- **Voice cloning legality** for GLaDOS-cloned voice (Valve IP). Personal use
  only; never ship.
- **Vector store earn-its-keep test** before v1 (synthetic 200-turn replay).
- **Backup / disaster recovery** for SQLite + keyring (offsite encrypted
  backup story).
- **Family member onboarding UX** — pairing a new client should be one short
  code, not editing config files.
