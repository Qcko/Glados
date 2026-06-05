# GLaDOS — project instructions

## Rubber-duck subagent for design and code changes

Rubber-ducking is the cheap fail-safe in this repo, and we lean on it. The cost
of a bug climbs by an order of magnitude at each stage it survives: a flaw
caught in **design** is cheaper than one caught while **coding**, which is far
cheaper than one that reaches **production**. So we duck at the two cheapest
places to catch one. A duck is an independent Agent (general-purpose, model
`opus`) that reviews and reports back but **never edits code**.

### 1. Design duck — before writing code, for slices with real design content

Before implementing a slice, rubber-duck the *plan* (not code) when the slice:

- introduces a new component, lifecycle, or state machine;
- adds a concurrency / async interaction (locks, tasks, ordering, races);
- crosses a trust or security boundary, or touches an `ARCHITECTURE.md`
  invariant;
- changes a schema, protocol, or other public contract; or
- has a shape not already dictated by an existing pattern in the repo.

Skip it when the slice just follows a settled pattern (wire a config field, add
a tool overlay, extend an existing loop). The design duck checks the intended
approach against the relevant `ARCHITECTURE.md` sections and probes for
shape-level flaws — the race, the missing state, the boundary leak — while they
are still a paragraph to rewrite, not code to unpick.

**Single duck vs. panel — match the reviewer count to the risk.** At design
stage the solution space is wide, so one perspective under-covers a high-risk
slice. For a *high-risk* design — one that crosses a **trust/security
boundary**, adds a **concurrency / async interaction**, or changes an
**`ARCHITECTURE.md` invariant** — spawn a *panel* of role-lensed reviewers in
parallel instead of a single duck. For a lower-risk design duck (an un-patterned
shape with no security or concurrency dimension), a single reviewer is enough.

Compose the panel from the lenses the slice actually needs — **the roster is
open, not a fixed committee.** Seat only lenses with real work to do, and add
whatever the slice calls for. Common lenses (illustrative, not exhaustive):

- **Architect** — fit with ARCH invariants, component boundaries, swappability,
  simplicity (is this the right shape, does it belong here).
- **Security** — `<external>` / untrusted-content discipline (§7), privacy
  (§9), the memory gate (§14), credentials, injection and trust-boundary leaks.
- **Concurrency & reliability (QA)** — races, lock/ordering, cancellation
  propagation (§6), failure modes, what's left untested.
- …and others as the slice demands — performance/latency, data migration,
  protocol/back-compat, UX, etc.

Give each lens the same neutral plan + ARCH pointers, but tell it to go deep on
its own class and assume the other lenses cover theirs (cuts overlap, forces
depth). Run them in parallel. Then **you** (the orchestrator) synthesize their
reports into one prioritized must-fix / should-fix / nit list — there is no
"lead" reviewer agent; synthesis is your job because you hold the conversation
context and act on the result. Explicitly surface any cross-lens conflict
(architect's "keep it simple" vs. security's "add the guard") for the user to
adjudicate — the panel's value is that tension, so don't paper over it.

### 2. Code duck — before committing, for any non-trivial change

After writing or editing code, before telling the user the slice is done and
before committing, duck the diff.

- Fires on any non-trivial Edit / Write under `src/`, `tests/`, or `configs/`,
  **tests included** — a test that encodes a wrong expectation locks a bug in
  as "correct", which is exactly what the duck exists to catch.
- Skip only genuinely trivial changes: documentation-only edits (`*.md`),
  config tweaks smaller than ~5 lines, formatting-only changes, version bumps,
  or a test that merely re-asserts already-verified behaviour with no new
  expectation.
- Once per coherent slice, not once per file. If a slice touches five files,
  one pass at the end covers them all.

### What to ask the duck

Self-contained prompt (the agent has no conversation context). Include:

- What the slice is meant to do, in one sentence.
- For a design duck: the intended approach and the key decision points /
  alternatives being weighed. For a code duck: the list of files changed with
  one-line summaries.
- Pointers to the relevant section of `ARCHITECTURE.md` so the duck can judge
  fit with the design.
- Specific concerns to probe (e.g. "can the reaper sleep a server a dispatch
  just woke?", "does this leak state across rooms?").
- Explicit instruction: report findings only, do not edit. Group by
  must-fix / should-fix / nit. Under 400 words.

### What to do with findings

- Must-fix: resolve before you proceed — revise the plan (design duck) or fix
  the code before committing (code duck).
- Should-fix: address now if cheap, otherwise note in the commit body and open
  a follow-up.
- Nit: ignore unless the user asks.

Briefly summarise the duck's findings to the user before you proceed (past the
plan, or to the commit) so they can override.
