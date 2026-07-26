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
