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

Two guards, both in `Organizer._run_tool_calls`, both ahead of the confirmation
gate (like the in-flight ledger: a call that is not being sent must not prompt
the room), and both standing down when `reply_language` is not English, because
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
    IF -- no --> G1{"quantity_arg set,<br/>value != 1,<br/>no count in utterance?"}
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

## Cue tables

`utterance.has_quantity_cue`: digits and number words (one..twenty, tens,
hundred, dozen) plus vague counts (couple, few, some, more, another, all...).
Word-bounded -- "one" lives inside "onion" and "phone". A digit glued to a unit
("1L", "7up") does not match and is not meant to: it names a product.
`has_repeat_cue`: another, again, more, extra, second, additional, further.
"more" is both, deliberately: "add more tomatoes" forwards with `repeat=true`
and lets the model's count through -- the user asked for more.

## Measurement

Not yet measured against a live model beyond the observed failure. What the
unit tests prove (`tests/test_write_guards.py`): the same add next turn is
refused; the model's `repeat` is overwritten; a repeat cue forwards with
`repeat=true`; a remove clears; a timeout answers `outcome_unknown`; the window
expires; non-English stands down; a refusal never sets `untrusted_seen`.
What they cannot: whether the model asks "how many" after `quantity_needed`,
or re-plans around `already_done` by switching to `add_to_cart(productId)`.

## Deliberately not built

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
