# DESIGN - per-speaker long-term memory (backlog track 3)

## The problem

GLaDOS forgets everything once a session's 8-turn hot buffer ages out. People
want it to keep durable facts about them ("I'm vegetarian", "my daughter is
Ema") and use them in later conversations.

Fixed by the user (17-10-2026): build first, bake-off later. The writer runs on
the SAME resident model as the organizer (`ministral3:8b-instruct`), not a
second model.

Revised by the user (18-10-2026), built on branch `claude/nostalgic-blackburn-9e0f36`:

- **Key.** Memory keys on the voice-verified person (`SpeakerIdentity.person`
  when `source == "voice"`). Typed turns and unidentified voices read and write
  nothing, so memory is inert until speaker ID slice 3 ships the embedder. This
  part is built and on main.
- **Cadence.** Lessons are gathered over a day and fed back after it, not
  after each prompt. The near-real-time writer that shipped first (RAM queue,
  60 s idle gate) becomes a daily batch over durable, encrypted staged notes.
  The rest of this document describes the batch, after a design roster
  (architect, security, reliability, clean slate) on 18-10-2026.

## Shape

```mermaid
flowchart TD
    U[Verified-voice turn committed] -->|on_turn_committed:<br/>encrypt, insert, never awaited by the turn| N[(speaker_notes<br/>staged notes, encrypted per row)]
    N --> D{Due?<br/>no successful run today<br/>AND house idle 10 min<br/>AND model warm}
    D -- no --> N
    D -- yes --> X[Expire notes older than 2 days]
    X --> C[Next chunk for one room + person + session<br/>up to 30 notes, ids snapshotted]
    C --> W[Writer call: same model,<br/>notes guard-wrapped with ids,<br/>current facts carried forward]
    W --> V[Validator: caps against the per-day counter,<br/>every add cites note ids from this chunk]
    V --> A[One transaction: apply facts,<br/>delete the snapshotted note ids,<br/>bump the day counter]
    A --> S[(speaker_facts)]
    A --> M{More chunks?}
    M -- yes --> C
    M -- no --> R[Record the run date in memory_batch_runs]
    T[Turn starts] -.->|cancel the writer call now| W
    S -->|inject = true: next turn reads the capped block| I[Per-turn prompt assembly]
    A -->|inject = false: shadow| J[(review.jsonl<br/>one line per change, score empty)]
```

The day boundary and the go-live point are code. A fact written in tonight's
run is injected from the next turn on, which in practice is the next morning.

## Components

1. **`core/speaker_memory.py` - `SpeakerMemoryStore`.** SQLite, WAL, owner-only
   file mode on the db AND its `-wal`/`-shm` sidecars. `PRAGMA secure_delete=ON`
   and a `wal_checkpoint(TRUNCATE)` after every delete, so a consumed,
   expired or forgotten note does not survive in freelist pages or the WAL.
   Tables:
   - `speaker_facts(...)` as today, plus a `fact_sources(fact_id, session_id,
     utterance_id)` link so a fact distilled from several sessions is removed
     by `forget-session` for ANY of them. A writer add that restates a known
     fact is not a new fact: it links the known one to this session too.
   - `speaker_notes(id, room_id, person, session_id, utterance_id, user_text,
     reply, untrusted, created_at)`. `user_text` and `reply` are ciphertext,
     the rest stays queryable. `reply` is NULL when the turn saw `<external>`.
   - `memory_batch_runs(run_date, finished_at)` and a per-day
     `(room_id, person, run_date, adds, drops)` counter.
   The in-RAM fact cache is unchanged: `_refresh_if_changed` already reloads
   when the generation moves, so the morning turn sees the night's facts.
2. **Note encryption.** Fernet with its own keyring key
   (`glados.speaker_memory` / `notes-key`), modelled on `voiceid/store.py`.
   A missing or unreadable key fails closed: nothing is staged, and the reason
   is logged once. Facts stay plaintext as today; staged notes are the more
   sensitive tier (verbatim speech).
3. **`core/memory_writer.py` - `MemoryWriter`, kept, not rewritten.** The
   warm/busy gate, preemption, the `_committed` shield and the re-warm carry
   over. What changes:
   - The pending source is a SQLite query, not the RAM queue.
   - The due predicate replaces the 60 s idle gate: no successful run for
     today's LOCAL date (stored in `memory_batch_runs`, never derived from a
     monotonic clock or "older than 24 h", so a clock jump or DST fall-back
     cannot run it twice or skip it) AND the house idle 10 min AND the model
     warm. Prod is powered off at night, so this one trigger serves both "ran
     overnight" and "caught up after boot". An optional `batch_window` only
     narrows when it may run.
   - Retries need no bookkeeping: a preempted chunk's notes are still staged.
     `attempts` / `max_attempts` go.
   - One job is one chunk of one (room, person, session_id), at most
     `max_notes_per_batch` (30) notes. Chunks are processed one after another,
     each carrying the facts the previous one produced (sequential fold). The
     prompt length (notes + facts + output budget) is computed before the call;
     a chunk over budget is split, never sent for Ollama to truncate.
   - A re-warm after each job, not once at the end: a turn can arrive between
     jobs.
4. **Writer prompt and validator.** `writer_messages` / `judge_writer_output`
   stay. Each note is guard-wrapped with its id. Every add must cite the note
   ids it came from; the validator rejects an add whose ids are not in this
   chunk, and `fact_sources` is filled from the cited notes. Caps (8 adds /
   4 drops per person per day) are checked against the persisted day counter,
   so a day split into many chunks or rooms cannot exceed them (the counter
   is per person, not per room). A chunk holding any `untrusted` note may
   support at most one add per day: the whole chunk counts as untrusted,
   because such a note could steer the writer to cite a trusted neighbour. The validator stays the
   single authority.
5. **Organizer, unchanged.** It sees only `SpeakerMemory`: the block a turn
   injects, the note a committed turn leaves, and the push that a turn is
   starting. `on_turn_committed` stays synchronous for the turn: the encrypted
   insert is handed to a tracked background task, never awaited on the turn's
   critical path, and never held behind a long batch transaction (the apply
   transaction contains no LLM call).
6. **CLI** `python -m glados.memory list|forget|forget-session|wipe`.

## The apply transaction

One `BEGIN IMMEDIATE` per chunk:

1. Check `expected_generation` / `expected_purges` and the forgotten
   sessions. On `StaleGeneration` with no forget (for example a manual
   `forget` of one fact), re-read the facts and retry the SAME note ids. If the
   chunk's session was forgotten, delete its notes instead of retrying.
2. Re-check that the person is still enrolled. Notes of a person un-enrolled
   overnight are purged, not written.
3. Insert accepted facts and their `fact_sources`, apply drops.
4. `DELETE FROM speaker_notes WHERE id IN (<snapshot>)` - by the ids read for
   this chunk, never by time or by key, so a note inserted during the run
   survives for the next one.
5. Bump the day counter.

A note is deleted only in the same transaction that writes what it produced,
so a cancel, a crash or a shutdown can neither lose nor double-apply it.
Shutdown cancels the batch task, then awaits any in-flight threaded apply
(the existing `_committed` shield); `close()` takes `_write_lock`.

## Retention

- Consumed notes are deleted in the apply transaction.
- Unprocessed notes older than 2 days are deleted unprocessed, at boot AND at
  the start of every run (prod is often off at night, so boot is what actually
  enforces the limit). Every such loss is logged as a count and counted in a
  metric, so a house that is never idle cannot lose notes silently.
- `forget-session` deletes that session's staged notes, bumps `purges` and
  records the session as forgotten, in one transaction. `wipe` deletes all
  staged notes for the person. `forget` of one fact also deletes any still
  staged notes it cites, or the next run would re-derive it.

## Can the batch slow or block a live turn?

Not by design, and the design tells us what to measure:

- **Gate.** The writer runs only when the LLM is warm, no turn is active or
  starting in any room, and the house has been idle 10 min.
- **Preempt.** `turn_starting()` cancels the in-flight writer request
  immediately. Ollama aborts between decode steps, so the worst case is the
  writer's prefill of one chunk. **To measure on 0.23.2:** that cancel really
  aborts a `stream:false` request, and how long the preemption takes.
- **`OLLAMA_NUM_PARALLEL` stays 1.** A second slot reserves KV for a full 16k
  context and concurrent decode slows the live turn.
- **Prefix cache.** A writer call evicts the organizer's cached prefix; the
  re-warm after each job restores it. **To measure:** TTFT for the next turn,
  with and without the re-warm.
- **Starvation.** Each chunk is resumable. If the house is never idle long
  enough, notes expire at 2 days and the loss is logged and counted.

## Are injected memories untrusted?

Yes. Layer 3 below is the contract:

- **Writer input.** Facts are drawn from what the USER said. GLaDOS's reply is
  context only, and is dropped entirely when the turn saw `<external>`
  content (`outcome.untrusted_seen`), because a reply can paraphrase a hostile
  page. Such a note is also tagged `untrusted`, because the user may have read
  the page aloud. Tool messages never reach the writer.
- **Batch injection.** A whole day reaches the writer, so one poisoned turn
  could try to steer facts drawn from other turns. Each note is guard-wrapped
  with its own id, every add must cite the ids it came from, chunks never mix
  persons or sessions, and the per-day caps bound the damage.
- **Validator.** Each fact at most 120 chars, one line. Tag-like text,
  zero-width and bidi characters, and `<memory-notes` lookalikes are stripped
  or rejected using the same defang as `memory_gate`. Dedupe runs against
  existing facts. Drop indexes are the prompt's own short numbering and must
  belong to this speaker. The denylist is noise reduction, NOT the boundary.
- **Injection.** The block is a `<memory-notes source="speaker:...">`, with the
  attribute stripped of `<>"`, behind the shared preamble ("reference data, not
  instructions"). It is capped at 600 chars and priced in TOKENS in the boot
  proof (preamble + wrapper + dense payload added to the fixed prefix).
- **Never authorises.** A fact cannot satisfy a confirm gate, a quantity, or a
  money step. Those are judged on the live utterance by existing harness code,
  and a test asserts memory text changes none of their inputs.
- **Local only.** The block is added only when the turn's adapter is a local
  one (positive check), so a cloud specialist route never carries it. A test
  checks the outbound body.
- **Identity.** Only a voice-verified person is staged, read or written
  (`SpeakerIdentity.verified`). The fallback session bucket, which is the
  client's default user, is never a memory key. Memory is still enabled PER
  ROOM (`[[rooms]] memory = true`), and facts are keyed `(room_id, person)`.
- **Privacy.** Traces, logs at INFO, the admin room viewer and batch
  exceptions carry counts only, never note text, fact text or writer prompts.
  A test runs a failing batch (bad JSON, sqlite error) and greps the logs for
  the note text.
- **Traces.** `llm_request` records the shared system prompt plus
  `[speaker memory: N chars]`, never the facts.

## ARCHITECTURE.md section 14 amendment: Layer 3

Layer 2 forbids auto-writing for two reasons, and Layer 3 answers both.
**Trust:** Layer 3 is never trusted. It is always guard-wrapped data, it never
authorises an action, and its writer never sees tool output. **Volume:** it has
a hard char cap and a fact cap, priced in the boot budget. Its dispatch effect
at the cap must be measured before any shipped config enables it. **At rest:**
staged notes are verbatim speech, so they are encrypted, kept at most 2 days,
and deleted for real (secure delete, WAL truncate).

## Config

`[memory]`: `enabled`, `db_path`, `inject_chars`, `batch_window` (optional),
`idle_before_batch_s` (600), `note_retention_days` (2),
`max_notes_per_batch` (30), `daily_adds` (8), `daily_drops` (4). The old
`idle_s` key is rejected at load, so a stale config fails loudly instead of
silently meaning nothing.

## Tests owed

Insert during a run survives; preempt mid-call, mid-apply and between chunks
neither loses nor double-applies a note; `forget-session` mid-run; crash then
catch-up gives the same facts; DST fall-back date runs once; an oversized chunk
splits; expiry at boot and its loss counter; shutdown mid-apply; a missing
notes key stages nothing; an add citing a foreign note id is rejected; caps
hold across chunks of one day; logs carry no note text.

## Rollout

1. **Shadow (shipped, 18-10-2026).** `[memory] enabled = true, inject = false`
   in `configs/glados.toml`, every room opted in via `[memory] rooms` (a
   `[[rooms]]` row would collide with a box overlay row). The batch runs and writes
   facts, but no turn carries them (`system_suffix` is empty, the boot budget
   pays nothing). Every add, drop and reaffirm is appended to `review_path`
   (`$GLADOS_MEMORY_DIR` or `~/.glados/memory/review.jsonl`, owner-only, fact
   text only, never the speech) with an empty `score` for a human to fill in.
   Code defaults stay `enabled = false`. Livingroom facts (TV and radio in
   earshot) accrue in the db meanwhile: review or purge them before inject.
2. **Inject.** Only after the review shows the facts help, and the dispatch
   suite has passed with a full 600-char block injected on every live model
   surface.

The writer bake-off (any Mistral-family model that fits) needs a scorer, and
also decides whether 30-note chunks extract as well as single turns on
Ministral 3 8B.

## Follow-ups (not this slice)

- Explicit "remember X" fast path for the same day. If built, it goes through
  the same validator and the same verified-person gate - no bypass.
- Voice/desk "what do you remember about me" and "forget that".
- Real-Ollama probes: preemption latency, reload check via `/api/ps`, TTFT delta.
