# STT corpus inputs

Inputs for the track 4 STT bake-off (`DESIGN-stt-fit.md`). Recordings are NOT
here: voice is biometric, so the corpus lives out of repo under
`<lab-root>/glados/stt_corpus/`, labelled `spk1..n`.

- `prompts.jsonl` -- 95 read prompts. `entities` are the slot words scored for
  entity error; tag `lex` = in the bias lexicon, `held` = deliberately not
  (~30 % of prompts), `num`/`time`/`unit`/`cmd` = scored, never biased.
  `{PERSON_A}`-style slots are filled from `lexicon.local.txt [people]`.
- `scenarios.txt` -- 15 cards for unscripted clips.
- `lexicon.example.txt` -- the frozen bias lexicon; copy to the gitignored
  `lexicon.local.txt` and add household names.
