# Demo scenarios

Walk-through script for the live-mic demo. Goal: validate the end-to-end
turn loop (mic → VAD → STT → LLM → TTS) and opportunistically eyeball
the other recently-landed slices.

> **Note:** STT is English-only by default (`distil-small.en`).
> Multilingual (EN + CS) input is deferred — recipe to re-enable lives
> in `configs/glados.toml` and ARCH §13.

## Setup

```powershell
ollama serve                    # in one terminal if not already running
uv run glados                   # in another; serves on http://127.0.0.1:8765
```

Open http://127.0.0.1:8765/ in a browser, grant mic permission, click the
mic button to start streaming. First boot downloads the Whisper model
(~250 MB) to `HF_HOME` (E:\hf) and pays a cold-start tax on the very
first LLM call (~10–15 s while Ollama lifts the model into VRAM).

Trace files land in `traces/` (per-session JSONL) — useful to grep after
the demo for any scenario where the in-browser feel was off.

---

## Scenarios

### 1. English smoke (baseline)
**Say:** "Hello, what time is it?"
**Expect:** STT transcribes correctly → LLM calls `time.now` → reply in
English with current time. **Pass condition:** complete turn, `done`
event in the browser, time tool fired (visible in browser as a
`tool_call`/`tool_result` block).
**Note:** First-turn-after-boot will be slower than steady-state.

### 2. Tool refusal
**Say:** "Use a tool called foobar that doesn't exist."
**Expect:** LLM either declines politely or calls a hallucinated name →
MCPRegistry returns "unknown tool" → LLM recovers and replies cleanly.
**Pass:** no crash; graceful recovery.

### 3. Barge-in
Ask a long question: "Tell me a five-paragraph story about a robot."
While GLaDOS is speaking, say: **"Stop."**
**Expect:** TTS stops within ~200 ms, `cancelled` event in browser,
queue clear. Barge-in regex covers `stop`, `cancel`, `halt`,
`nevermind`, etc.

### 4. TTS feedback gate (self-trigger prevention)
Set browser volume up. Ask anything that produces a long reply. While
GLaDOS is speaking, **say nothing** — but let the speaker → mic loop
attempt to retrigger.
**Expect:** no new turn starts from GLaDOS's own audio. The 200 ms
post-Done cooldown should also drop any echo tail.
**If a turn does self-trigger:** note the speaker setup (built-in vs.
external) — the cooldown may need bumping per ARCH §3 / SESSION.md
open thread.

### 5. Multi-turn trace append
Open the WebSocket, run **three** turns in a row (don't disconnect
between). Then inspect the trace file:
```powershell
ls traces\ | sort LastWriteTime | select -last 1
```
Open it and confirm **all three turns' events are present** (the
`TraceWriter` 'a'-mode fix). Each turn should have its own
`turn_start` / `user_text` / `tts_chunk` / `done` block. **Fail
condition:** only the last turn's events visible → 'a' mode regressed.

### 6. Untrusted content (`<external>` defense)
Call any tool whose result the LLM will echo. The toy server doesn't
embed a prompt-injection payload by default, so this scenario is more
of a "look at the trace and confirm" rather than an active probe:
inspect the trace JSONL for a `tool_result` event; confirm the content
ended up wrapped in `<external>...</external>` before being handed to
the LLM. **The full prompt-injection test lives in
`tests/test_untrusted_content.py`** — this demo step is just an
eyeball that the wrapping survives a real round-trip.

### 7. Latency feel (steady-state)
After warmup (scenario 1 already ran), do **five short English turns
in a row** ("hello", "what time is it", "roll a die", "tell me a
joke", "thanks"). Note subjective latency for each. **Loose target:**
turn 2 onwards should feel sub-2-second from end-of-speech to first
TTS chunk. If turn 2 is much slower than turn 5, the warmup story is
incomplete. If everything is uniformly slow on CPU, the LLM is the
bottleneck (try a smaller Ollama model or GPU offload).

---

## After the demo

Record in SESSION.md:
- Which scenarios passed / failed.
- Subjective steady-state latency (one number, e.g. "~1.5 s p50, ~2.5 s
  p95 turn-to-first-chunk on CPU int8").
- Any surprises worth a follow-up slice (e.g. gate cooldown too short,
  self-trigger heard).
