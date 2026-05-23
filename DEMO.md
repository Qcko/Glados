# Demo scenarios

Walk-through script for the live-mic demo. Goal: validate the
multilingual STT swap (commit `f1f2045`, ARCH §13) end-to-end, and
opportunistically eyeball the other recently-landed slices.

## Setup

```powershell
ollama serve                    # in one terminal if not already running
uv run glados                   # in another; serves on http://127.0.0.1:8765
```

Open http://127.0.0.1:8765/ in a browser, grant mic permission, click the
mic button to start streaming. First boot after the STT swap downloads
the multilingual `small` model (~450 MB) to `HF_HOME` (E:\hf) — expect a
30–60 s pause on the very first transcription.

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

### 2. Czech smoke (the swap's headline)
**Say:** "Ahoj, kolik je hodin?"
**Expect:** STT auto-detects Czech and transcribes it. LLM still calls
`time.now`. **Reply must be in English.**
**Pass conditions:** (a) Czech transcript visible in the browser, (b)
reply in English, not Czech. **If the reply is in Czech, the SYSTEM_PROMPT
clause isn't doing its job — flag it.**

### 3. Czech non-tool turn
**Say:** "Řekni mi vtip o programátorech." ("Tell me a joke about
programmers.")
**Expect:** Czech transcribed, English-language joke spoken back.
**Pass:** the LLM understood the Czech intent and produced an English
response. **Watch for:** any partial Czech in the reply — the
"always-English" clause should hold even for non-tool turns.

### 4. Mixed-language session
Within one open WebSocket, alternate:
- "What's two plus two?" (toy.add)
- "Kolik je tři krát čtyři?" ("What's three times four?")
- "Roll two dice." (toy.roll_dice)

**Expect:** each turn auto-detects independently, tool calls fire,
replies all in English. **Watch for:** any context bleed between turns
(e.g. the model latching onto Czech after one Czech utterance).

### 5. Tool refusal under multilingual
**Say:** "Použij nástroj, který neexistuje, jmenuje se foobar."
("Use a tool that doesn't exist, called foobar.")
**Expect:** LLM either declines politely (English) or calls a
hallucinated name → MCPRegistry returns "unknown tool" → LLM recovers
and replies in English. **Pass:** no crash; English-language recovery.

### 6. Barge-in (English only — known limitation)
Ask a long question: "Tell me a five-paragraph story about a robot."
While GLaDOS is speaking, say: **"Stop."**
**Expect:** TTS stops within ~200 ms, `cancelled` event in browser,
queue clear. **Known limitation:** barge-in regex is English-only
(`stop`, `cancel`, `halt`, `nevermind`, etc.). Czech "přestaň" will
**not** trigger barge-in today — that's expected, not a regression.
File as a follow-up if it matters.

### 7. TTS feedback gate (self-trigger prevention)
Set browser volume up. Ask anything that produces a long reply. While
GLaDOS is speaking, **say nothing** — but let the speaker → mic loop
attempt to retrigger.
**Expect:** no new turn starts from GLaDOS's own audio. The 200 ms
post-Done cooldown should also drop any echo tail.
**If a turn does self-trigger:** note the speaker setup (built-in vs.
external) — the cooldown may need bumping per ARCH §3 / SESSION.md
open thread.

### 8. Multi-turn trace append
Open the WebSocket, run **three** turns in a row (don't disconnect
between). Then inspect the trace file:
```powershell
ls traces\ | sort LastWriteTime | select -last 1
```
Open it and confirm **all three turns' events are present** (the
`TraceWriter` 'a'-mode fix). Each turn should have its own
`turn_start` / `user_text` / `tts_chunk` / `done` block. **Fail
condition:** only the last turn's events visible → 'a' mode regressed.

### 9. Untrusted content (`<external>` defense)
Call any tool whose result the LLM will echo. The toy server doesn't
embed a prompt-injection payload by default, so this scenario is more
of a "look at the trace and confirm" rather than an active probe:
inspect the trace JSONL for a `tool_result` event; confirm the content
ended up wrapped in `<external>...</external>` before being handed to
the LLM. **The full prompt-injection test lives in
`tests/test_untrusted_content.py`** — this demo step is just an
eyeball that the wrapping survives a real round-trip.

### 10. Latency feel (steady-state)
After warmup (scenario 1 already ran), do **five short English turns
in a row** ("hello", "what time is it", "roll a die", "tell me a
joke", "thanks"). Note subjective latency for each. **Loose target:**
turn 2 onwards should feel sub-2-second from end-of-speech to first
TTS chunk. If turn 2 is much slower than turn 5, the warmup story is
incomplete. If everything is uniformly slow, `small` on CPU int8 may
be too heavy — options are `int8_float16`, GPU device, or pinning
`whisper_language` to skip auto-detect when known.

---

## After the demo

Record in SESSION.md:
- Which scenarios passed / failed.
- Subjective steady-state latency (one number, e.g. "~1.5 s p50, ~2.5 s
  p95 turn-to-first-chunk on CPU int8").
- Any surprises worth a follow-up slice (e.g. Czech reply leaked
  through, gate cooldown too short, self-trigger heard).
