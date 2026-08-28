"""Deterministic per-turn outcome classification.

A turn can end three ways from the harness/router's point of view:

- ``done`` -- the model did what was asked (or answered conversationally).
- ``needs-user`` -- the model handed back with a question without acting.
- ``failed`` -- a tool error went unrecovered, or the tool loop blew its budget.

The classification is derived from *observable* tool results, never from the
model's self-report. A drifting local model narrates failure cheerfully (see
the 2026-05-30 bake-off T6: two tool errors followed by an upbeat clarifying
question), so its narration is worthless as a completion signal. The wire
already carries the truth -- each dispatch records ``ok`` -- this module just
aggregates it to the turn level so the v2.6 router can use it as an
escalation input.

Pure error-derivation misses *semantic* drift, where a tool returns 200 OK but
the model never acts on the result (bake-off T2: ``search_products`` succeeded,
the model just never added to the cart). The lightweight goal-check below
closes the common case of that gap without leaving the deterministic lane: when
the user issued an *action* request ("add milk", "remove the eggs") and the
turn ran tools but never landed a successful *mutating* call, the action did
not happen -- that is drift, not success. It stays a heuristic on observable
signals (the user's verb + the per-tool ``mutating`` flag), not a model
self-report or a semantic judgement of the reply text.

Reading the user's verb is NOT done here -- `core/utterance.py` owns that, and
the organizer passes the answer in as ``TurnRecord.action_intent``. This module
sees only the record of what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

TurnOutcomeKind = Literal["done", "needs-user", "failed", "confabulated"]


@dataclass(frozen=True)
class ToolRecord:
    """One tool dispatch within a turn, as the registry observed it."""

    tool: str  # qualified name, e.g. "dunnes.add_to_cart_by_name"
    ok: bool
    # Side-effecting calls (cart writes, checkout, login, money). Set by the
    # organizer as `spec.mutating or spec.requires_confirmation`: a gated tool
    # always mutates, but confirmation alone is not sufficient -- the Dunnes
    # cart writes are un-gated side effects and carry `mutating` directly.
    #
    # Both halves are per-deployment OVERLAY, declared per tool name in
    # `servers.toml` and applied after a real MCP `tools/list` (the wire schema
    # has no slot for trust flags). It is not derived from anything, so it is
    # only as complete as that file: a side-effecting tool nobody flagged reads
    # here as a read, and every guard keyed on "did a mutation land" stays
    # asleep for it. Widening what counts as an action request widens what this
    # flag has to be right about -- check the overlay before leaning on it.
    mutating: bool = False
    # Lowercased free-text argument values -- what the call was ABOUT
    # ("milk", "bananas"). Ids and numbers are excluded: they never appear in a
    # spoken reply, so they cannot corroborate or contradict one. Used only by
    # the claim check below, which asks whether the thing the reply says was
    # changed is the thing any successful call actually touched.
    subjects: tuple[str, ...] = ()
    # The call timed out after it was sent, so whether it landed is unknown.
    # `ok` stays False -- the turn really did fail, and saying otherwise would
    # report success for a write that may never have happened -- but every
    # guard asking "is it safe to run this again" must treat it as a maybe.
    indeterminate: bool = False


@dataclass
class TurnRecord:
    """Mutable accumulator a turn fills in as it runs, then classifies once."""

    tools: list[ToolRecord] = field(default_factory=list)
    final_text: str = ""
    loop_exhausted: bool = False
    # True when the user's request was an imperative to change external state
    # (set by the organizer via core/utterance.is_action_request). Drives the
    # goal-check: such a turn is only `done` if a mutating call actually landed.
    action_intent: bool = False
    # True once this turn has ingested `<external>` bytes, seeded from whether
    # the session's retained history already holds some. Drives the
    # confirmation gate in `Organizer._run_tool_calls`, which is where the
    # reasoning lives.
    #
    # Seeded per turn rather than recomputed, so note what the retry paths
    # depend on: scope fallback, specialist escalation and `_finish_the_job`
    # each mint a fresh record and replay the PRIOR history, so a retry
    # correctly inherits the session flag without the failed attempt's bytes.
    # A retry changed to replay the failed attempt's messages would need this
    # flag carried across with them.
    untrusted_seen: bool = False

    # Mutating calls this turn sent but never got an answer to, keyed by
    # `(server.name, canonical args)`. Re-dispatching one of these would be the
    # duplicate write the whole design exists to prevent, so the organizer
    # answers it from here instead of putting it back on the wire. Per turn,
    # because a fresh turn is a fresh decision by the user.
    in_flight: set[tuple[str, str]] = field(default_factory=set)

    def record_tool(
        self,
        tool: str,
        ok: bool,
        *,
        mutating: bool = False,
        indeterminate: bool = False,
        args: dict | None = None,
    ) -> None:
        self.tools.append(
            ToolRecord(
                tool=tool,
                ok=ok,
                mutating=mutating,
                indeterminate=indeterminate,
                subjects=_subjects(args),
            )
        )

    def made_successful_mutation(self) -> bool:
        """True if a side-effecting tool call is KNOWN to have landed this turn.

        The goal check's predicate: did the thing the user asked for actually
        happen. Deliberately not the replay predicate -- see
        `may_have_mutated`, which asks the different question."""
        return _has_successful_mutation(self.tools)

    def may_have_mutated(self) -> bool:
        """True if external state might already have changed this turn.

        "Did the goal get achieved" and "is it safe to do this again" are
        different questions, and every replay path wants this one. A mutating
        call that timed out is recorded `ok=False`, so asking
        `made_successful_mutation` there answers "nothing landed, safe to
        replay" about a cart write that may be executing as it answers."""
        return any(t.mutating and (t.ok or t.indeterminate) for t in self.tools)


def classify(turn: TurnRecord) -> TurnOutcomeKind:
    """Reduce a finished turn to a single typed outcome.

    Priority order matters: an unrecovered tool error is ``failed`` even when
    the turn ends on a question (T6 ended on a cheerful question *after* two
    errors -- that is a failure, not a clarification request)."""
    if turn.loop_exhausted:
        return "failed"
    if _has_unrecovered_error(turn.tools):
        return "failed"
    if said_nothing(turn):
        return "failed"
    if _confabulated(turn) or claimed_a_change_it_did_not_make(turn):
        # Ahead of the drift check on purpose. A turn can be both drifted AND
        # making a false claim, and only `confabulated` gets the reply replaced
        # and kept out of history (see Organizer._handle_confabulation);
        # `failed` would leave the false "Milk removed from cart." both spoken
        # and committed, which is the poisoning this whole module exists to
        # stop. The two are otherwise disjoint -- `_confabulated` needs a
        # zero-tool turn, `_action_drifted` needs tools -- so nothing else
        # changes order here.
        return "confabulated"
    if _action_drifted(turn):
        # Asked to act, ran tools, but never landed the mutation. Ending on a
        # question is a hand-back (the model punted instead of acting);
        # ending on a statement is silent drift (bake-off T2: searched,
        # narrated the JSON, never added). Both mean the action didn't happen.
        return "needs-user" if _ends_on_question(turn.final_text) else "failed"
    if _ends_on_question(turn.final_text) and not _has_successful_call(turn.tools):
        return "needs-user"
    return "done"


def said_nothing(turn: TurnRecord) -> bool:
    """The turn ended with no reply text at all. Every other guard here asks
    whether the model did the RIGHT thing; this one asks whether it answered,
    and an unanswered turn cannot be `done` however clean its tool calls were.

    Observed 2026-08-25 on qwen3:4b: a reasoning model spent its entire
    num_predict budget on `thinking` tokens and was cut at `done_reason=length`
    before emitting a single content token. Tools had all succeeded and nothing
    errored, so the turn fell through to `done` -- the harness reported success
    for a turn the user heard nothing from. Silence is the one failure the user
    cannot tell from a crash, so it must not classify as success."""
    return not turn.final_text.strip()


def _confabulated(turn: TurnRecord) -> bool:
    """The user asked to mutate state, the turn dispatched *no tools at all*,
    yet the reply declares the action done ("Sure, adding that now.", "Added
    milk to your cart."). That is a fabricated completion -- the signature of a
    poisoned history steering the model away from tool-calls (SESSION 2026-06-15:
    six consecutive zero-dispatch turns all narrating success). It is the
    zero-tool case `_action_drifted` deliberately leaves alone: there, a
    *declarative* ending is the discriminator from an honest hand-back. Ending on
    a question is a clarification ("which milk?") and stays `needs-user`."""
    return (
        turn.action_intent
        and not turn.tools
        and bool(turn.final_text.strip())
        and not _ends_on_question(turn.final_text)
    )


# Past-tense assertions that a change HAPPENED. Present and future forms are
# absent on purpose ("I'll add", "to add", "shall I remove") -- those are
# intentions, and only a completed claim can be a false one.
# The claim verbs themselves, excluded from a clause's distinctive words: they
# say that something changed, never WHAT changed, so they can neither
# corroborate nor contradict a dispatch record.
_CLAIM_WORDS = frozenset(
    {"added", "removed", "deleted", "cleared", "emptied", "updated", "changed", "set"}
)

_CLAIM_RE = re.compile(
    r"\b(added|removed|deleted|cleared|emptied|updated|changed)\b"
    # "set ... to 2" is a quantity claim; "set to expire in 30 minutes" is not,
    # so the number is required rather than optional.
    r"|\bset\b[^.!?]{0,20}?\bto\s+\d",
    re.IGNORECASE,
)

# A claim inside one of these is not this turn reporting its own work:
# a denial, a report of what someone else did, or a reference back to an
# earlier turn. Each was a real false positive before it was excluded.
_NOT_THIS_TURNS_DOING = re.compile(
    r"\b(not|n't|nothing|never|no)\b"  # "I have not added it", "Nothing was removed"
    r"|\balready\b|\bearlier\b|\bpreviously\b|\blast night\b"  # a prior turn
    r"|\bby\s+the\s+\w+",  # "was updated by the store"
    re.IGNORECASE,
)

# Words too common to prove a reply is talking about a given tool argument.
_UNDISTINCTIVE = frozenset(
    {
        "a", "an", "and", "the", "to", "of", "for", "in", "on", "at", "it",
        "my", "your", "cart", "item", "items", "one", "two", "some", "all",
        "please", "thing", "things", "product", "products",
        # Cart-meta nouns. A reply routinely names one right after a real
        # change ("Added the eggs and updated your basket."), and no tool
        # argument ever contains one -- so judging a claim by them accuses a
        # turn that did exactly what it said. Added 25-08-2026 after four
        # verified false positives of that shape.
        "total", "order", "basket", "trolley", "list", "quantity", "qty",
        "everything",
    }
)


def claimed_a_change_it_did_not_make(turn: TurnRecord) -> bool:
    """The reply says something was added/removed/set, and the dispatch record
    does not support it.

    This is the sibling of `_confabulated` for turns that DID call tools -- and
    it is the one that fires in practice, because it never consults
    `action_intent`. Measured 25-08-2026: four of five real mutation requests
    ("Actually, add eggs instead", "...and then remove the milk", "Take one of
    the milks off") do not match `is_action_request` at all, so every guard
    keyed on it stays asleep. The reply is where the claim lives, and the
    dispatch record is the ground truth; neither depends on how the user
    phrased the request.

    Two real failures from the same day's bake-off, both previously `done`:
    the model removed the bananas, never called add(eggs), and said "Eggs added
    to cart"; and it called view_cart, removed nothing, and said "Milk removed
    from cart".

    It FAILS OPEN wherever it cannot judge, because a false positive replaces a
    correct spoken reply with a canned "no record of that" line -- telling a
    user their shopping did not happen when it did is its own kind of lie."""
    if _ends_on_question(turn.final_text):
        # "Shall I have the milk removed?" is an offer, not a report.
        return False
    claims = [c for s in _sentences(turn.final_text) for c in _claim_clauses(s)]
    if not claims:
        return False

    # A mutating call that timed out may well have made the change the reply
    # describes, so it excuses a claim exactly as a successful one does --
    # but only a claim about ITS OWN subject. Bailing on the whole turn
    # instead would let one outstanding call excuse every invented claim
    # beside it, which is the failure `_claim_clauses` exists to prevent.
    landed = [t for t in turn.tools if t.mutating and (t.ok or t.indeterminate)]
    if not landed:
        # Claimed a change with nothing side-effecting behind it at all.
        return True

    subject_words = {w for tool in landed for s in tool.subjects for w in _words(s)}
    if not subject_words:
        # Every successful call identified its target by an id, a number or an
        # enum -- none of which a spoken reply repeats. Nothing comparable, so
        # do not accuse. Note this tests for usable WORDS, not merely for the
        # presence of arguments: a call with only `state="on"` has arguments and
        # still gives us nothing to check against.
        return False
    return any(_claim_unsupported(c, subject_words) for c in claims)


def _claim_clauses(sentence: str) -> list[str]:
    """The individually-checkable claims inside one sentence.

    A sentence is subdivided because the check used to union every landed
    call's subjects and test the reply as a whole, which made ONE true claim
    excuse every false one beside it. Measured 25-08-2026: "Eggs added and the
    milk removed." with only the add landed returned False -- the word "eggs"
    matched, so the invented removal went unchallenged. Splitting on the
    coordinators lets the milk clause be judged on its own.

    Only clauses that themselves assert a change are returned, so a conjoined
    object ("I added the eggs and the milk") yields a single claim rather than
    a second, verbless clause that no call could ever support."""
    if not _asserts_a_change(sentence):
        # Negation and prior-turn references are judged on the whole sentence,
        # deliberately: "I added the eggs but did NOT remove the milk" reads as
        # one disclaimed report, and splitting first would strip the "not" from
        # the clause it governs.
        return []
    return [c for c in re.split(r"\band\b|\bbut\b|\bthen\b|[;,]", sentence)
            if _asserts_a_change(c)]


def _claim_unsupported(clause: str, subject_words: set[str]) -> bool:
    """True if this one claim names something no successful call touched.

    The claim verbs are stripped before comparing, so a bare "Added." carries
    no distinctive words and FAILS OPEN. Without that it would compare
    {"added"} against the subjects of a real add and accuse a turn that did
    exactly what it said."""
    named = {
        w
        for w in _words(clause) - _CLAIM_WORDS
        # `_subjects` keeps only values containing a run of letters, so a
        # bare number is absent from `subject_words` BY CONSTRUCTION.
        # Judging a claim by one ("set the quantity to 250") could only
        # ever accuse.
        if not w.isdigit()
    }
    return bool(named) and not (named & subject_words)


def asserts_a_change(text: str) -> bool:
    """True if any sentence reports a completed change. Public so the organizer
    can spot the inverse case -- a turn that really did mutate something but
    whose reply matches nothing in the vocabulary below. Those replies are the
    only reliable source of phrasings the vocabulary is missing."""
    return any(_asserts_a_change(s) for s in _sentences(text))


def _asserts_a_change(sentence: str) -> bool:
    return bool(
        _CLAIM_RE.search(sentence) and not _NOT_THIS_TURNS_DOING.search(sentence)
    )


def _sentences(text: str) -> list[str]:
    # Judged per sentence so one honest clause cannot excuse a false one, and
    # a negation in a neighbouring sentence cannot excuse a claim in this one.
    return [s for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _UNDISTINCTIVE and len(w) > 2
    }


def _subjects(args: dict | None) -> tuple[str, ...]:
    """The free-text values a call was aimed at.

    Identifiers are excluded even though they arrive as strings: a spoken reply
    names the product, never its id, so keeping them made every id-based call
    look like a contradicted claim. The test is "contains a run of letters",
    not "is not all digits" -- `prod-123` and a UUID are just as absent from
    speech as `100806893`, and only Dunnes happens to use bare digits.

    A call carrying a list or dict argument yields nothing at all. Those are
    the batch flows (a recipe's worth of items), where the top-level strings
    describe the request rather than any one item, and judging a twelve-item
    add by the word "lasagne" would be worse than not judging it."""
    if not args:
        return ()
    if any(isinstance(v, (list, dict, tuple)) for v in args.values()):
        return ()
    return tuple(
        value
        for v in args.values()
        if isinstance(v, str)
        and (value := v.strip().lower())
        and re.search(r"[a-z]{3}", value)
    )


def _action_drifted(turn: TurnRecord) -> bool:
    """The user asked to mutate state, the turn ran at least one tool, yet no
    successful mutating call landed. A pure read/search can't satisfy an action
    request. Turns with no tool calls are out of scope here -- the search-and-
    narrate drift this targets always shows tool activity; the zero-tool
    fabricated-completion case is handled by `_confabulated`."""
    return (
        turn.action_intent
        and bool(turn.tools)
        and not _has_successful_mutation(turn.tools)
    )


def _has_successful_mutation(tools: list[ToolRecord]) -> bool:
    return any(t.ok and t.mutating for t in tools)


def _has_unrecovered_error(tools: list[ToolRecord]) -> bool:
    """An error on a tool is recovered only by a *later* successful call to
    the same tool. A success that precedes the error does not count."""
    for name in {t.tool for t in tools}:
        calls = [t for t in tools if t.tool == name]
        last_error = _last_index(calls, ok=False)
        if last_error is None:
            continue
        recovered_after = any(c.ok for c in calls[last_error + 1 :])
        if not recovered_after:
            return True
    return False


def _last_index(calls: list[ToolRecord], *, ok: bool) -> int | None:
    for i in range(len(calls) - 1, -1, -1):
        if calls[i].ok is ok:
            return i
    return None


def _has_successful_call(tools: list[ToolRecord]) -> bool:
    return any(t.ok for t in tools)


def _ends_on_question(text: str) -> bool:
    return text.rstrip().endswith("?")
