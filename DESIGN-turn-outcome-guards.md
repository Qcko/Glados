# DESIGN — the turn-outcome guards, and the gap left open

## The problem

A turn can end by doing what was asked, or by any of several failures that the
model narrates as success. `core/turn_outcome.py` classifies the turn from the
*observable dispatch record* rather than the model's self-report, because a
drifting local model narrates failure cheerfully and its narration is worthless
as a completion signal.

Four guards had accumulated there. This pass asked whether a fifth could catch
**honest under-delivery** — the model doing the first half of a compound
instruction and truthfully reporting what it found, without doing the second
half:

> **User:** "Show me what's in my cart and then remove the milk."
> **GLaDOS:** calls `view_cart`, says "Cart has 1 x 3L milk.", stops.
> The milk is never removed. Outcome: `done`.

The answer is that it cannot, with the signals available. That negative result
is the main content of this document — see *Deliberately not built*.

## The flow

```mermaid
flowchart TD
    turn["turn ends<br/>(tool records + reply text)"] --> exhausted{"tool loop<br/>blew its budget?"}
    exhausted -- yes --> failed["failed"]
    exhausted -- no --> err{"a tool error<br/>never recovered?"}
    err -- yes --> failed
    err -- no --> silent{"no reply text<br/>at all?"}
    silent -- yes --> failed
    silent -- no --> lie

    subgraph lies["the two lie-detectors — these alone replace the reply"]
        lie{"zero tools, yet<br/>declares the action done?"}
        claim{"a claim the dispatch<br/>record cannot support?"}
        lie -- no --> claim
    end

    lie -- yes --> confab["confabulated<br/>reply replaced, kept OUT of history"]
    claim -- yes --> confab
    claim -- no --> drift{"asked to act, ran tools,<br/>nothing mutated?"}

    drift -- "yes, ends on a question" --> needs["needs-user"]
    drift -- "yes, ends on a statement" --> failed
    drift -- no --> ask{"ends on a question<br/>with no successful call?"}
    ask -- yes --> needs
    ask -- no --> done["done"]

    gap["HONEST UNDER-DELIVERY<br/>half the ask done, reply TRUE<br/>no guard fires -- lands here"]:::gap
    gap -.-> done

    classDef gap fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#000
```

The dotted edge is the point of the diagram: every guard above is *correctly*
silent on that input, and it arrives at `done` with nothing wrong that any of
them can see.

## Why each guard misses it

- **The claim check** needs a claim to corroborate. The reply is true.
- **The drift check** needs `action_intent`, which is anchored on a *leading*
  imperative — and the utterance opens with a read verb.
- **The confabulation retry** only fires on `confabulated`, which this is not.

## Ordering that matters

The two lie-detectors sit ahead of the drift check on purpose. A turn can be
both drifted *and* making a false claim, and only `confabulated` gets the reply
replaced and kept out of history; `failed` would leave "Milk removed from cart."
both spoken and committed, which is the poisoning the module exists to stop.

`claimed_a_change_it_did_not_make` never consults `action_intent`, which is why
it is the one that fires in practice — the reply is where the claim lives and
the dispatch record is ground truth, and neither depends on how the user phrased
the request.

## Invariants the implementation holds

1. **Fail open wherever it cannot judge.** A false positive replaces a correct
   spoken reply with a canned "no record of that" line *and* rewrites committed
   history — telling a user their shopping did not happen when it did is its
   own kind of lie. False negatives are cheap by comparison; the asymmetry is
   deliberate and every check is built to it.
2. **Claims are judged one at a time.** The check used to union every landed
   call's subjects and test the reply as a whole, so one true claim excused a
   false one beside it: "Eggs added and the milk removed." with only the add
   landed returned clean. Each claim clause is now judged alone.
3. **Claim verbs, cart-meta nouns and bare numbers never decide a claim.**
   They say *that* something changed, never *what*, and no tool argument ever
   contains them — so a claim naming only those has nothing checkable in it and
   declines to judge. Without this, "Added the eggs and updated your basket."
   accuses a turn that did exactly what it said.
4. **The `^` anchor on the action heuristic is the entire safety argument.**
   Whole utterances rarely *begin* with an action verb by accident. Everything
   the heuristic is allowed to do must preserve that property. The heuristic
   itself lives in `core/utterance.py` (`is_action_request`) — it reads the
   user's request, not the turn's record, and the organizer passes its answer
   in as `TurnRecord.action_intent`.
5. **Nothing replays after a successful mutation.** Every re-drive path checks
   it, or the side effect (cart write, checkout, send) fires twice.
6. **Four drives per utterance, enforced.** Base drive, capability fallback,
   specialist escalation, finish-the-job. `tests/test_drive_ceiling.py` pins it
   per brain — 2 on the primary, 2 on the specialist — so a fifth path cannot be
   added silently.

## Deliberately not built

**Detecting honest under-delivery.** Two mechanisms were designed, measured and
rejected on 25-08-2026.

**Splitting the utterance on coordinators** and testing each clause for action
intent. This was the obvious fix and it is the wrong one. The heuristic is safe
*because* of the `^` anchor; splitting manufactures new string-starts, and 11 of
the action verbs have an ordinary non-imperative reading in exactly that
position — "…and **change** is due on Friday", "…and **buy** one get one free
deals", "…and **order** should be under fifty euro". It trades a bounded miss
for an unbounded false positive, in the direction that costs most. It also does
not pay for itself: it left three of the five known misses unfixed, and no regex
separates "add milk **and** bread" (one action, conjoined object) from "show the
cart **and** remove the milk" (two clauses) — that distinction is
part-of-speech, not pattern.

Measured on 1372 logged utterances the split produced zero false positives, and
that number should not be trusted: the corpus is a scripted test suite, not
conversation, and contains almost none of the constructions above. It cannot
refute the finding; it only shows those phrasings are absent from data that was
never conversational.

**A per-clause satisfaction ledger**, requiring each action clause to have a
matching successful mutating call. Its matcher fails on the example that
motivated it: "take one of the **milks** off" against a call carrying
`"3L milk"` does not overlap (no stemming) and produces a *false accusation*,
while "remove the small milk" against a call that removed the large one *does*
overlap and is wrongly satisfied — too strict on morphology and too loose on
identity at once. It is also blind on every batch flow, where under-delivery is
most likely, because a list argument yields no subject words at all. Recovery
was unavailable to it regardless: in the partial case a mutation has already
landed, so invariant 5 forbids the replay.

**What would change the answer:** an operation axis on the tool contract
(add/remove/set, so a clause can be matched against what a call *did* and not
only what it was *about*), or naturalistic logged utterances to measure against
rather than a scripted suite. Neither exists today. Note the first widens the
adapter contract that ARCHITECTURE §6 keeps deliberately tiny, and that is an
invariant change to argue on its own terms, not a classifier tweak.

Until then the compound instruction is unguarded, and the honest reply reaches
the user with half the job done.
