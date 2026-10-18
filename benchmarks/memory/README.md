# Memory-writer bake-off corpus

Inputs for the memory-writer bake-off (`DESIGN-memory-scorer.md`).

- `cases.jsonl` -- synthetic cases only (`origin: "synthetic"`, enforced by a
  test). No real household names, speech or facts belong here. Real-derived
  cases are human paraphrases kept out of repo under
  `<lab-root>/glados/memory_corpus/cases.local.jsonl`.
- `split` is frozen: `test` cases decide the bake-off and are never
  adjudicated; only `dev` cases get `also_ok` / `forbid` added after a run.
- Matchers are `{"all": [...], "any": [...], "none": [...]}` over lowercase
  words. `none` keeps negations and other people from counting as a hit.
- A case with no `expect_add` and no `expect_drop` is an empty day: any add is
  a false add. At least 40 % of the corpus should be empty days, the common
  real day.

Run: `uv run python -m glados.memory bakeoff --model <tag>` with GLaDOS
stopped (it refuses to run while :8765 answers).
