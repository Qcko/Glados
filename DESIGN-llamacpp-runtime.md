# DESIGN -- the llama.cpp runtime question, and why it is DEFERRED

Status: **recommendation, awaiting sign-off.** Two decisions inside are marked
SIGN-OFF and are not mine to make. No migration code exists.

## The question

GLaDOS runs its local LLM through Ollama's `/api/chat`. Should it run through
llama.cpp's `llama-server` (`/v1/chat/completions`) instead?

It reached the design roster because the shipped model, `ministral3:8b-instruct`,
emits tool calls as PROSE -- `[TOOL_CALLS]name[ARGS]{json}` -- which Ollama does
not parse, so every call arrived as SPOKEN TEXT. Making that model shippable at
all took a bespoke parser (`brain/llm/tool_text.py`), a `text_tool_format` key
and its pairing validator, a buffered-streaming hold-back (`_filtered`/`_drain`),
an `LLMToolCall.from_text` flag feeding a widened confirmation gate, and a
hand-ported Ollama template (`configs/ministral3-8b-instruct.Modelfile`),
because mistralai's own GGUF ships a single-turn template with no `.Tools` and
no `range .Messages`.

A spike (27-08-2026) showed `llama-server` serving the SAME GGUF blob with
mistralai's canonical `chat_template.jinja` via `--jinja` returns structured
`tool_calls` for every case that forced that work. Adopting it would delete all
of it.

## The recommendation: DEFER, and take the runtime-independent half now

Measurement removed both of the migration's quantitative arguments and its
qualitative one, while leaving its real prize intact but smaller than it looked.

**What was claimed, and what measurement found:**

| claim | finding |
| --- | --- |
| Better tool-calling | Equivalent. Dispatch scores match once samplers are pinned (below). |
| Faster / better on the R9700 | Latency-neutral today. All three runtimes within ~10%. |
| Removes a US layer (provenance) | Theatre, and net-NEGATIVE on supply chain. |
| No bespoke parser in the trust path | FALSE. The parser moves into llama.cpp, looser and unobservable. |

Against that, what it genuinely buys is real: deleting ~575 lines of
prose-parsing that exists only to work around a gap llama.cpp does not have, no
hand-ported template, grammar-constrained decoding, and two vendor-agnostic
backends to choose between on the R9700.

The judgement is that the prize is worth having and the moment is wrong. The
security work required to migrate SAFELY (below) is comparable in size to what
the migration deletes, and it buys no measured improvement in latency or
dispatch quality. Doing it now spends that budget for a net-zero result.

**Revisit trigger, concrete:** the R9700 32 GB arriving AND Mistral Small 24B
being qualified on it. That is where llama.cpp's advantages stop being
theoretical -- a model that does not fit 12 GB, a card where the vendor-agnostic
Vulkan build matters, and a template-fidelity question that the hand-ported
Modelfile route answers badly at 24B.

**Take now, regardless of runtime:** the two guards in "The guard that is
actually at stake" below are worth having on the Ollama path today. They are
not migration work.

## Evidence

All artifacts out-of-repo under `<lab-root>/glados/ministral/`:
`PROBES_llamacpp_roster.md`, `BENCH_runtime_3way.md`,
`RESCORE_llamacpp_dispatch.md`, `RP_CONFIRM_ollama.md`.

**Latency -- 3-way, each runtime alone on the card (27-08-2026).** Deep prompt
(~4.1-4.4k, the depth that matters; the assembled prompt plateaus near 4.7k) on
Ministral 8B: Ollama 4337 t/s prompt / 105.3 t/s generation; llama.cpp CUDA
4452 prompt; llama.cpp Vulkan 4053 prompt. Every arm within ~10% at both depths
on both shipped-tier models. The predicted 20-40% Vulkan prompt-processing
penalty DID NOT REPLICATE -- Vulkan is 7-9% behind CUDA on Ministral and 3%
AHEAD on qwen3:4b. So the R9700 portability story costs almost nothing, and
Ollama is not the slow option.

**Dispatch quality -- re-scored through the same gold-label harness.** With
`repeat_penalty` pinned to Ollama's 1.1, llama.cpp reproduces the Ollama h2 row
EXACTLY, false fires included. At h0 the sampler changes nothing and llama.cpp
is one fixture behind, under the harness's own ~4-fixture noise floor. **The
runtimes are equivalent; the sampler was the entire difference.** That finding
has already shipped as `16fe1a1`, which sends `repeat_penalty` explicitly
instead of inheriting a value that is 1.1 on Ollama and 1.0 on llama.cpp.

NOT re-scored: the 22-point human-judged quality scorecard, which needs a
GLaDOS llama.cpp adapter and human scoring. Prose quality is where
`repeat_penalty` could still matter. **Dispatch equivalence is not evidence
about prose.**

## The guard that is actually at stake

This is the part the spike got wrong, and it is the reason the roster ran.

The spike claims adoption means "no bespoke parser in the trust-sensitive path,
and no route from the spoken channel to a dispatch." Both halves are false. A
scan of `llama-server-impl.dll` finds the `[TOOL_CALLS]` marker,
`parse_tool_calls`, and `common_chat_peg_builder::python_style_tool_calls`, with
`--jinja` DEFAULT-ENABLED. The parser does not disappear -- it moves inside
llama.cpp, where GLaDOS cannot see or gate it.

Three guards exist today. They do not all have the same fate:

1. **Rule 3, the offered-tools allowlist -- SURVIVES, for free.** Both paths
   already enforce it. `_events_from_chunk` looks the wire name up in
   `name_map` and routes a miss to `server="unknown"`, which MCPRegistry
   answers with "unknown tool". An unoffered name is not dispatchable on either
   path. (An earlier reading of the security review said this had to be
   re-implemented; reading the code says otherwise.)
2. **Rule 2, "the marker must START the reply" -- LOST.** `parse_tool_calls`
   in `tool_text.py` refuses a call with narration in front of it, precisely
   because a mid-reply marker is what an echo of `<external>` content looks
   like. llama.cpp is built to accept content-then-calls, because Mistral emits
   both. **Measured, not inferred:** a probe asking for one sentence of
   preamble then a call returned `content` AND a structured `tool_calls` with
   `finish_reason: "tool_calls"`.
3. **`from_text` and the widened confirmation gate -- LOST unless replaced.**
   Text-recovered calls are currently gated harder because they came from a
   channel that also carries untrusted bytes. Under llama.cpp that provenance
   still exists; it just becomes invisible.

**The attack rule 2 was blocking.** `dunnes` is `untrusted = true`. A
seller-authored product title, review, or Q&A field carries the literal text
`[TOOL_CALLS]dunnes__add_to_cart_by_name[ARGS]{...}`. It enters `messages` and
persists across up to 7 further tool-loop passes. If the model echoes it
mid-reply: today rule 2 refuses it, and `from_text and spec.mutating` forces a
confirmation the Dunnes cart tools otherwise skip
(`requires_confirmation = false`, deliberately). After migration it arrives
looking identical to a genuine call and dispatches un-confirmed.

**Mitigating, and explicitly not a defence:** the planted injection was refused
4/4 by Ministral at temperature 0, in two phrasings. That is model behaviour
under one prompt shape, not a guard.

**Therefore, and independent of the runtime decision:** replace `from_text`
with a broader and better-founded gate -- require confirmation for a MUTATING
call whenever `<external>`-wrapped content is present in this turn's `messages`.
It covers the structured path too, which `from_text` never did, and it closes
the gap SESSION.md already records ("the `from_text` gate catches only the
mutating subset"). This is worth doing on the Ollama path now.

## The flow, and where each guard sits

```mermaid
flowchart TD
    U["Untrusted tool result<br/>wrapped &lt;external&gt;"] --> CTX["messages<br/>persists up to 7 passes"]
    CTX --> M["Model generates"]

    M --> OLL["OLLAMA today<br/>calls arrive as PROSE"]
    M --> LCP["LLAMA.CPP proposed<br/>parsed inside llama-server"]

    OLL --> R2{"rule 2<br/>marker STARTS reply?"}
    R2 -->|"no - an echo looks like this"| SPOKEN["Refused<br/>stays spoken text"]
    R2 -->|yes| FT["from_text = True"]

    LCP --> NOR2["No rule-2 equivalent<br/>content-then-call ACCEPTED"]
    NOR2 --> STRUCT["from_text = False"]

    FT --> MAP{"rule 3<br/>offered this turn?"}
    STRUCT --> MAP
    MAP -->|no| UNK["server = unknown"]
    MAP -->|yes| GATE{"Confirmation gate"}

    GATE -->|"from_text AND mutating"| CONFIRM["Ask the user"]
    GATE -->|otherwise| DISPATCH["Dispatch"]

    STRUCT -.->|"THE HOLE"| DISPATCH
    GATE -.->|replace this arm| NEW{"PROPOSED not built<br/>MUTATING and &lt;external&gt; this turn?"}
    NEW -->|yes| CONFIRM

    style SPOKEN fill:#E6F2EC,stroke:#2E7D5B,color:#12171C
    style UNK fill:#E6F2EC,stroke:#2E7D5B,color:#12171C
    style CONFIRM fill:#E6F2EC,stroke:#2E7D5B,color:#12171C
    style NOR2 fill:#F8EAE7,stroke:#A6402F,color:#12171C
    style DISPATCH fill:#F8F0DF,stroke:#9A6A18,color:#12171C
    style NEW fill:#E8F1F6,stroke:#2F6F8F,color:#12171C
```

Read it as: everything reaching `MAP` is already past the only guard that
differs between the runtimes. The dotted edge from `STRUCT` to `DISPATCH` is
the hole -- a structured call carrying no provenance marker, against tools that
opt out of confirmation.

## If the decision is to migrate anyway -- the requirements

Not optional, and not deferrable to "later in the migration". Sources:
security, concurrency and performance lenses, all confirmed by probe where
marked.

**Trust boundary**
- Replace `from_text` with the `<external>`-present gate above. Do not simply
  delete it.
- `--api-key` (value in the OS keyring per ARCH section 9, never in
  `glados.toml`), `--cors-origins localhost`, `--no-cors-credentials`,
  `--no-webui`. CONFIRMED from `--help`: `--cors-origins` defaults to `*` with
  `--cors-credentials` ENABLED, which is browser-reachable and a regression
  against Ollama. The server prints its own warning about this at startup.
- Build an explicit minimal `env=` for the launcher. `LLAMA_ARG_TOOLS` can wire
  `exec_shell_command` / `write_file` into the server from the environment, and
  `ollama_lifecycle.py`'s `Popen` inherits the full environment today.
- Hash-pin the binary AND the Jinja template, fail closed on mismatch, mirroring
  the ARCH section 14 memory gate. `llama-server.exe` is NotSigned and is a 9 KB
  launcher over `llama-server-impl.dll` in a user-writable directory, outside
  LocalGuard. `/props` returns the full template text, so verifying it at
  runtime is cheap.
- NOT required: `--no-slots` is worth passing but is not urgent. The
  transcript-leak finding DID NOT REPLICATE on b10645 -- `/slots` returns only
  `id`, `n_ctx`, `speculative`, `is_processing`.

**Lifecycle and reliability**
- Add a non-`CancelledError` failure arm to `_run_user_text`. Today a
  mid-stream backend death escapes to the room-queue worker, which logs and
  moves on, so the room gets NO terminal frame -- no `Done`, no speech, no
  outcome. Ollama's daemon surviving bad requests is what keeps this path cold;
  a per-model process makes it live.
- Readiness must verify model IDENTITY via `/props`, not just `/health`.
  Adopting a live port the way `OllamaLifecycle` does would let an orphaned
  server serve the PREVIOUS model for a whole session with correct-looking logs.
  The ladder is CONFIRMED: 503 `{"status":"loading model"}` then 200
  `{"status":"ok"}`.
- Assert real GPU offload after start and refuse to serve otherwise. `-ngl 99`
  SILENTLY fell back to CPU during probing (7.4 tok/s, no error) because Ollama
  held 5.84 GB via `keep_alive="-1"`. In a voice assistant that presents as
  "GLaDOS got slow", not as a fault.
- `-np 1` QUEUES over-capacity requests rather than rejecting them, so a second
  room's turn is indistinguishable from a slow one and waits out the 600 s read
  timeout. `room_queues.py`'s "Ollama serialises" docstring becomes false.
- Windows Job Object with kill-on-close, and kill by PID not image name -- two
  `llama-server.exe` are indistinguishable by name.

**Config and budget**
- `num_ctx` stops being enforceable: it is a launch flag (`-c`, divided across
  `--parallel` slots), not a per-request field. Read the real window from
  `/props` and warn on mismatch, or the context-pressure check compares against
  a number nobody enforces.
- `max_tokens` must be sent per request AND `--n-predict` set server-side.
  CONFIRMED: both default to `-1` = unbounded, which would un-bound the 2026-06-18
  repetition loop.
- Send `stream_options: {"include_usage": true}` or token accounting goes dark,
  taking the context-pressure, truncation and burned-budget warnings with it.
- Re-derive the coupling invariant
  (`max_assembled_prompt + num_predict <= num_ctx`) on the new template. Peak
  prompt measured 4369 vs Ollama's 3908 at h0 (+11.8%) and 7793 vs 7330 at h2
  (+6.3%): mistralai's canonical template is more verbose, so every existing
  margin is 6-12% optimistic.
- Own the GGUF directory. Pointing `-m` at Ollama's blob store is not a
  migration -- both tags share model blob `sha256-33e7a72c...` and differ ONLY
  in their template layer (142-byte HF single-turn vs our 497-byte tool-capable
  port), so reading the blob silently drops the template the Modelfile exists
  to supply.

## SIGN-OFF 1 -- one backend or two

The architect and migration lenses disagree, and the disagreement is real.

- **Architect:** the `num_predict = 512` scar was resolved by REMOVING the
  unexercised surface, not by keeping two. Carrying both local backends
  recreates the condition unless CI genuinely exercises both.
- **Migration:** the scar was about an UNEXERCISED surface, not a second one.
  Both are safe exactly as long as the contract tests are parameterised over
  both -- but an undated "keep both for now" is how it reproduces.

They converge on dual-with-a-kill-date and differ on whether that is a
concession or the plan. **Recommendation: moot under DEFER.** If the migration
proceeds later, land llama.cpp alongside with a written deletion date and
parameterised contract tests, and treat a missed date as a decision to revert
rather than to extend.

## SIGN-OFF 2 -- provenance

This one cuts against the reasoning that drove the model swap of 27-08-2026, so
it deserves an explicit decision rather than a silent drop.

The provenance case for llama.cpp is that ggml is Bulgarian-origin while Ollama
is a US company. Judged on merits it does not hold: Ollama SHIPS the same
`ggml-*.dll`, so the inference engine is identical; the ggml team joined Hugging
Face, a US company, in February 2026; and the measurable supply-chain delta is
NEGATIVE -- a signed installer with a content-addressed model store, traded for
an unsigned zip fetched from a browser and living outside LocalGuard.

**Recommendation: drop provenance from the justification entirely and migrate,
if at all, on the tool-parsing merits.** Note this does NOT retroactively
question the MODEL choice: Ministral's weights are Mistral's regardless of which
runtime loads them, and that swap also stands on a 20/22 score. It questions
only whether provenance argues for changing RUNTIME, where it does not.

## Deliberately not built

- **A `llamacpp` adapter.** Nothing in the repo; the only implementation is a
  lab-only harness used to score the runtime
  (`<lab-root>/glados/llamacpp_llm.py`), which exists to measure the runtime and
  is explicitly NOT a migration.
- **Two-server primary/specialist topology.** Priced and rejected for 12 GB:
  10.0 GB against ~9.7 GB usable, and `-ngl 99` turns that into a load failure
  or a silent CPU fallback rather than Ollama's graceful spill. Viable on the
  R9700; revisit there. `local_smart_model = ""` aliases the specialist to the
  primary today, so no distinct local specialist has ever actually run.
- **Process supervision of a bundled binary.** If the migration proceeds, prefer
  requiring the user to run `llama-server` and keeping only probe +
  `wait_until_ready`, rather than growing `ollama_lifecycle.py` into a model
  registry wearing a lifecycle costume.

## Open, and worth knowing

- The 22-point quality suite is unmeasured on llama.cpp (see Evidence).
- `ARCHITECTURE.md` section 5's LLM row is ALREADY false -- it claims vLLM +
  Qwen2.5-32B/Llama-3.3-70B while the system ships Ollama + Ministral 8B. Any
  runtime decision, including this deferral, should correct it.
- Section 6's "All adapter calls accept a cancellation token" is also already
  false: `OllamaLLM.chat` takes no token, and interrupt works via task
  cancellation. It propagates correctly; the sentence does not describe it.
- `README.md` is a migration behind, still naming Ollama + Qwen3 and telling a
  new operator to `ollama pull qwen3:8b`.
