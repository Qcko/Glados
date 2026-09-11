# DESIGN -- write guards: a count the user never said, and an add that already landed

## The problem

Observed 11-09-2026 on `ministral3:8b-instruct` against the live Dunnes server.
Turn 1, "add tomatoes to the cart": `add_to_cart_by_name(query="tomatoes")`,
one pack added. Turn 2, the same sentence: the model called
`remove_from_cart(productId=...)` and then
`add_to_cart_by_name(query="tomatoes", quantity=4)`. Net cart: four packs, and
the four came from nowhere.

Neither existing guard could fire. The Dunnes server refuses an *identical*
additive write inside two minutes, but the model rewrote the arguments, so the
second write was not identical. The organizer's in-flight ledger
(`DESIGN-dispatch-cancellation.md`) refuses a re-issue *within a turn*; these
were two turns. And the project rule stands: small local models follow prompts
unreliably, so the fix is code in the harness, not a line in the system prompt.

Three guards (the third added after a re-run, invariant 10), all in
`Organizer._run_tool_calls`, all ahead of the confirmation gate (like the in-flight ledger: a call that is not being sent must not prompt
the room), and all standing down when `reply_language` is not English, because
the cue tables cannot read anything else.

## The flow

```mermaid
flowchart TD
    U["utterance<br/>(last user message)"] --> A
    M["model emits a mutating call"] --> A
    A["align repeat flag<br/>repeat := has_repeat_cue(utterance)<br/>(overwrites whatever the model sent)"]
    A --> B["broadcast ToolCall + trace<br/>(args as they will go to the wire)"]
    B --> IF{"in-flight this turn?"}
    IF -- yes --> R0["_ALREADY_ATTEMPTED<br/>(existing)"]
    IF -- no --> G3{"removes with these args,<br/>utterance only asks to add?"}
    G3 -- yes --> R3["refuse: not_removed<br/>ok=True, satisfied=False"]
    G3 -- no --> G1{"quantity_arg != 1, or count_arg<br/>under an add-only utterance,<br/>and no count in utterance?"}
    G1 -- yes --> R1["refuse: quantity_needed<br/>ok=True, satisfied=False"]
    G1 -- no --> G2{"additive and<br/>ledger has the key<br/>inside the window?"}
    G2 -- no --> C["confirm gate -> dispatch"]
    G2 -- yes --> RC{"repeat cue<br/>in utterance?"}
    RC -- yes --> C
    RC -- no --> R2["refuse: already_done<br/>or outcome_unknown<br/>ok=True, satisfied=certain"]
    C --> O{"result"}
    O -- "additive, ok or indeterminate" --> L["ledger.note(key, quantity, certain=ok)"]
    O -- "non-additive on the same server,<br/>ok or indeterminate" --> X["ledger.clear(session, server)"]
    R1 --> T
    R2 --> T
    R3 --> T
    L --> T
    X --> T
    T["tool message to the model<br/>(refusals unwrapped: GLaDOS wrote them)"]
```

Key = `(server.tool, canonical args)` with the tool's `quantity_arg` and the
server's `repeat` dropped, string values lowercased and whitespace-collapsed.
The in-flight ledger uses the same canonicaliser with nothing dropped.

## Invariants the implementation holds

1. **Guard 1 runs before guard 2.** Otherwise the ledger answers "already done"
   to `add(tomatoes, quantity=4)`, the `satisfied` record excuses the model's
   "added four", and the invented count is laundered into a true claim.

2. **`repeat` is the user's word, never the model's.** Every call whose schema
   carries `repeat` has it overwritten from `has_repeat_cue(utterance)`. A model
   that sets `repeat: true` on its own would otherwise defeat both this ledger
   (a new key) and the server's own identical-write refusal. Rewritten before
   the `ToolCall` broadcast and the trace event, so the confirm dialog and
   `traces/` show the call that went to the wire.

3. **The ledger holds only harness-authored facts**: tool, the model's
   arguments, the clock, the quantity, and whether the result was certain.
   Never the server's result content. That is what lets a refusal be delivered
   outside any `<external>` wrapper (section 7): GLaDOS wrote every byte of it.

4. **A refusal is `ok=True`, `mutating=False`.** `ok=True` like the intercom
   refusal, so the turn is not `failed` and does not escalate to the specialist
   (which would re-issue the same call into the same refusal). `mutating=False`
   because nothing changed this turn, so `may_have_mutated` and the recheck
   escape hatch stay truthful. The goal-check reads a third flag instead:
   `ToolRecord.satisfied`, set only for a *certain* `already_done`, counts as
   landed in `_has_successful_mutation` and in the claim check's `landed`
   list. On the record, not the turn, so one satisfied tomato call cannot
   excuse "and I added milk too".

5. **`quantity_needed` meets nothing.** `satisfied=False`: the goal-check sees
   an action turn with no landed mutation, so a model that asks "how many?"
   classifies `needs-user` and one that asserts "added four" is caught by the
   claim check exactly as before.

6. **Any non-additive write on the same server clears that server's entries
   for the session.** A remove is by `productId`, an add by free-text `query`;
   they cannot be matched, and after a remove the ledger's "already in the
   cart" may be false. Clearing is coarse and fails open toward the server's
   own refusal. Scoped to the server because `room.speak_into` and a timer are
   mutating too and say nothing about the cart -- an intercom message between
   two adds must not re-arm the duplicate. A call that answered `ok` without
   taking effect (the intercom's refusal) clears nothing. Today's sequence now
   plays: the remove clears, the `quantity=4` add is refused by guard 1, and a
   later plain "add tomatoes" adds one.

7. **A timed-out add is ledgered too**, `certain=False`, and answered
   `outcome_unknown` ("may already be in the cart; check") rather than
   `already_done`. Stamped when the result arrives, not when the call was
   sent -- Dunnes timeouts are 400 s, longer than the window.

8. **`additive` and `quantity_arg` are overlay fields** (`servers.toml`), so
   no server's tool or argument names live in code. Set on the four Dunnes
   adds; `quantity_arg` only on the two with a count, because for
   `set_*`/`adjust_*` a 1 or a -1 is a real request.

9. **Bounded like `_history`**: per session, LRU-evicted with it, pruned to
   the window on every touch, dropped on start-over.

10. **Guard 3: an add request does not remove** (added 11-09-2026, after the
    post-fix re-run). Turn 2 opened with `remove_from_cart` again; that
    non-additive write cleared the ledger (invariant 6), the re-add went to the
    wire, the SERVER refused it as a duplicate, and the cart ended empty while
    GLaDOS said "already in the cart". Guard 3 runs first: when the utterance
    opens with add / put / buy / order (`utterance.is_add_request`, anchored
    like `is_action_request`) and carries no removal cue anywhere, a write
    flagged `removes` is refused with `not_removed`, `ok=True`,
    `satisfied=False`. Being a local answer it clears nothing, so the re-add
    meets the ledger and "already in the cart" is true. Naming the argument
    makes the flag conditional: `delta_arg` on `adjust_*` removes only at
    `delta <= 0`, so "add one more milk" via `adjust(delta=1)` still goes
    through; `count_arg` on `set_*` removes only at `quantity <= 0`. An
    argument that is not a number counts as removing.

11. **An absolute set answering a count-free add is a guessed quantity.**
    `set_cart_quantity_by_name(milk, 1)` from three takes two out, and the
    harness cannot see the cart to know. But "add milk" named no number, so
    any value a `count_arg` tool carries is invented: guard 1 refuses it with
    `quantity_needed` (a note pointing at the add tool, since a set has no
    count-free form). Only when the utterance is add-only -- "remove the milk"
    has no count either, and its `set(quantity=0)` must go through -- and not
    when it carries a count cue ("add two milk"). A repeat cue alone ("add
    milk again") does not license it: it says on top, not how many, and the
    add tool already carries `repeat` for exactly that.
    A `delta_arg` tool is exempt: a delta of one is "add one", not a guess.

## Cue tables

`utterance.has_quantity_cue`: digits and number words (one..twenty, tens,
hundred, dozen) plus vague counts (couple, few, some, more, another, all...).
Word-bounded -- "one" lives inside "onion" and "phone". A digit glued to a unit
("1L", "7up") does not match and is not meant to: it names a product.
`has_repeat_cue`: another, again, more, extra, second, additional, further.
"more" is both, deliberately: "add more tomatoes" forwards with `repeat=true`
and lets the model's count through -- the user asked for more.
`is_add_request`: leading add / put / buy / order, and none of remove, delete,
take, replace, instead, swap, change, switch, less, fewer, reduce, drop, out,
off, only, down, minus, without, rid, zero, none, empty, clear, cancel, undo,
set, update, make, no, not, ditch, scrap, forget, and the digit 0. Broad on
purpose: a missed cue refuses a remove the user wanted; a false cue only falls
back to the behaviour before guard 3.

## Measurement

Not yet measured against a live model beyond the observed failure. What the
unit tests prove (`tests/test_write_guards.py`): the same add next turn is
refused; the model's `repeat` is overwritten; a repeat cue forwards with
`repeat=true`; a remove clears; a timeout answers `outcome_unknown`; the window
expires; non-English stands down; a refusal never sets `untrusted_seen`.
What they cannot: whether the model asks "how many" after `quantity_needed`,
or re-plans around `already_done` by switching to `add_to_cart(productId)`.

## Deliberately not built

- **A set to a lower non-zero count under a counted or vague-count
  utterance.** "Add two milk" licenses `set(milk, 2)`, and "add some milk" or
  "add more milk" licenses `set(milk, 1)`; either reduces a cart holding three.
  The harness cannot see the cart; invariant 11 covers only the count-free add.
  A repeat word ("again", "second") is deliberately not a count.
- **A remove-only refusal still escalates.** It meets no goal, so the turn is
  `failed` and the specialist re-drives it; the re-issue goes back through
  guard 3 and is refused again (tested). A wasted pass, not a removal -- the
  same cost guard 1 already pays.
- **A harness-authored reply for "removed, then the shop refused the re-add".**
  Planned alongside guard 3, dropped: guard 3 removes the path that produced
  it under an add-only utterance, and detecting the case means matching the
  server's own refusal text -- untrusted content steering a control decision.

- **Cross-tool re-add.** `add_to_cart_by_name(query)` then
  `add_to_cart(productId)` for the same product are different keys. Guard 1
  still catches the invented count; the rest is the server's refusal.
- **Account-scoped ledger.** Sessions are per `(room, speaker)`; the cart is
  per account. "Add tomatoes" from the desk and again from the kitchen mic is
  not caught.
- **Window measured from the utterance.** It is measured from the check. A
  long Selenium call inside turn 2 could push past it.
- **Per-language cue tables.** Off-English both guards stand down.
- **Reading the count from an earlier turn.** The cue check reads only the
  last user message. "Shall I add four for the recipe?" / "yes" /
  `add(quantity=4)` is refused and costs one round-trip ("how many?" /
  "four") on a well-behaved dialogue. Scanning history for the count would
  let a scraped product name supply it, so the utterance stays the only
  source.
