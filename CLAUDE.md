# GLaDOS — project instructions

## Rubber-duck subagents (design roster + code duck)

The general discipline now lives in the global rules **"Design roster — parallel
role-lensed review before building (design duck)"** and **"Rubber-duck
non-trivial code (code duck)"** in `~/.claude/CLAUDE.md` (model policy: `fable`
first, `opus` fallback, always the newest version via the unversioned alias --
never a pinned dated model id, per the global *Subagent model choice* section). Read
those for the trigger, the dynamic-roster mechanics, aggregation, and how to
handle findings. This section only records the **GLaDOS-specific
specializations**.

### Check the plan against `ARCHITECTURE.md`

GLaDOS has an `ARCHITECTURE.md` with numbered invariants; the design roster's job
here is to judge fit against it. Give each lens the relevant section pointers,
and map lenses to sections:

- **Architect** — component boundaries, swappability, fit with ARCH invariants.
- **Security** — `<external>` / untrusted-content discipline (§7), privacy (§9),
  the memory gate (§14), credentials, injection and trust-boundary leaks.
- **Concurrency & reliability (QA)** — races, lock/ordering, cancellation
  propagation (§6), failure modes, what's left untested.

Add other lenses as a slice demands (performance/latency, data migration,
protocol/back-compat, UX).

### When to seat the full panel

Beyond the global trigger, treat any slice that **crosses a trust/security
boundary**, **adds a concurrency/async interaction**, or **changes an
`ARCHITECTURE.md` invariant** as high-risk: seat the full role-lensed panel
rather than a single reviewer, because at design stage one perspective
under-covers a high-risk slice.

### Probe examples

When prompting a duck, include GLaDOS-shaped concerns to probe, e.g. "can the
reaper sleep a server a dispatch just woke?" or "does this leak state across
rooms?". Point the duck at the relevant `ARCHITECTURE.md` section so it can judge
fit with the design.

## Model testing: always run BOTH shipped-tier models

Whenever a change is measured against a model -- bake-off suites
(`scripts/bakeoff_run.py`), dispatch scoring, prompt changes, a regression
probe -- run it against **both `qwen3:8b` and `qwen3:4b`**, never only the
shipped one. `qwen3:8b` is what `configs/glados.toml` ships; `qwen3:4b` is the
hardcoded default in `core/config.py`, so a config omitting `model` runs it.
Both are live surfaces and a result from one is not evidence about the other.

Measuring only the shipped model is how the `num_predict = 512` bug survived:
it was harmless on the non-reasoning incumbent and silently blanked every qwen3
turn, and nothing caught it because nothing exercised the other model.

Restart GLaDOS between models -- room history carries across runs and a killed
run's turns get replayed into the next one (see the confabulation trigger).
