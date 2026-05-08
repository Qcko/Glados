# GLaDOS — project instructions

## Rubber-duck subagent for code changes

After writing or editing code in this repo — but before telling the user the
slice is done — spawn an independent Agent (general-purpose, model `opus`) to
rubber-duck the change. The agent does **not** modify code; it reviews and
reports back.

### When to fire

- Any non-trivial Edit / Write to files under `src/`, `tests/`, or
  `configs/`. Skip for: documentation-only edits (`*.md`), config tweaks
  smaller than ~5 lines, formatting-only changes, version bumps.
- Once per coherent slice, not once per file. If a slice touches five files,
  one rubber-duck pass at the end covers them all.
- Before committing. The duck's findings inform whether the commit is ready
  or needs revision.

### What to ask the duck

Self-contained prompt (the agent has no conversation context). Include:

- What the slice was meant to do, in one sentence.
- The list of files changed with one-line summaries.
- Pointers to the relevant section of `ARCHITECTURE.md` so the duck can
  judge fit with the design.
- Specific concerns to probe (e.g. "is the session keying correct given
  §3?", "does this leak state across rooms?").
- Explicit instruction: report findings only, do not edit. Group by
  must-fix / should-fix / nit. Under 400 words.

### What to do with findings

- Must-fix: address before committing.
- Should-fix: address now if cheap, otherwise note in the commit body and
  open a follow-up.
- Nit: ignore unless the user asks.

Briefly summarise the duck's findings to the user before the commit so they
can override.
