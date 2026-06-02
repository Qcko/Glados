"""Deterministic per-turn outcome classification.

A turn can end three ways from the harness/router's point of view:

- ``done`` — the model did what was asked (or answered conversationally).
- ``needs-user`` — the model handed back with a question without acting.
- ``failed`` — a tool error went unrecovered, or the tool loop blew its budget.

The classification is derived from *observable* tool results, never from the
model's self-report. A drifting local model narrates failure cheerfully (see
the 2026-05-30 bake-off T6: two tool errors followed by an upbeat clarifying
question), so its narration is worthless as a completion signal. The wire
already carries the truth — each dispatch records ``ok`` — this module just
aggregates it to the turn level so the v2.6 router can use it as an
escalation input.

Pure error-derivation misses *semantic* drift, where a tool returns 200 OK but
the model never acts on the result (bake-off T2: ``search_products`` succeeded,
the model just never added to the cart). The lightweight goal-check below
closes the common case of that gap without leaving the deterministic lane: when
the user issued an *action* request ("add milk", "remove the eggs") and the
turn ran tools but never landed a successful *mutating* call, the action did
not happen — that is drift, not success. It stays a heuristic on observable
signals (the user's verb + the per-tool ``mutating`` flag), not a model
self-report or a semantic judgement of the reply text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

TurnOutcomeKind = Literal["done", "needs-user", "failed"]

# Imperative verbs that imply the user wants the assistant to *change* external
# state this turn, not just look something up. Matched at the start of the
# request (after an optional politeness/filler lead-in) so a mid-sentence
# mention ("tell me what to add") doesn't trip the action heuristic. Read verbs
# (show, list, find, what/which/is) are deliberately absent — a read request is
# satisfied by a successful search, so it must not demand a mutating call.
_ACTION_VERBS = (
    "add",
    "remove",
    "delete",
    "buy",
    "order",
    "book",
    "set",
    "change",
    "update",
    "cancel",
    "clear",
    "empty",
    "put",
    "schedule",
    "send",
    "play",
    "turn",
)
_ACTION_INTENT_RE = re.compile(
    r"^(?:(?:please|hey|ok(?:ay)?|glados|can you|could you|would you|i(?:'d| would) like (?:you )?to|i want (?:you )?to)[\s,]+)*"
    rf"(?:{'|'.join(_ACTION_VERBS)})\b",
    re.IGNORECASE,
)


def is_action_request(text: str) -> bool:
    """True if the user's utterance reads as an imperative to mutate external
    state ("add milk to the cart") rather than a question or a read ("what's
    in my cart?"). Used by the goal-check to demand a successful mutating tool
    call before calling such a turn ``done``."""
    return bool(text) and _ACTION_INTENT_RE.match(text.strip()) is not None


@dataclass(frozen=True)
class ToolRecord:
    """One tool dispatch within a turn, as the registry observed it."""

    tool: str  # qualified name, e.g. "dunnes.add_to_cart_by_name"
    ok: bool
    # Side-effecting calls (cart writes, checkout, login, money). Mirrors
    # ToolSpec.requires_confirmation — the existing per-tool flag for
    # "this mutates external state".
    mutating: bool = False


@dataclass
class TurnRecord:
    """Mutable accumulator a turn fills in as it runs, then classifies once."""

    tools: list[ToolRecord] = field(default_factory=list)
    final_text: str = ""
    loop_exhausted: bool = False
    # True when the user's request was an imperative to change external state
    # (set by the organizer via is_action_request on the user text). Drives the
    # goal-check: such a turn is only `done` if a mutating call actually landed.
    action_intent: bool = False

    def record_tool(self, tool: str, ok: bool, *, mutating: bool = False) -> None:
        self.tools.append(ToolRecord(tool=tool, ok=ok, mutating=mutating))

    def made_successful_mutation(self) -> bool:
        """True if a side-effecting tool call landed this turn. The cloud
        escalation path checks this before re-running a `failed` turn: a turn
        that already mutated external state must not be replayed cold, or the
        side effect (cart write, checkout, send) fires twice."""
        return _has_successful_mutation(self.tools)


def classify(turn: TurnRecord) -> TurnOutcomeKind:
    """Reduce a finished turn to a single typed outcome.

    Priority order matters: an unrecovered tool error is ``failed`` even when
    the turn ends on a question (T6 ended on a cheerful question *after* two
    errors — that is a failure, not a clarification request)."""
    if turn.loop_exhausted:
        return "failed"
    if _has_unrecovered_error(turn.tools):
        return "failed"
    if _action_drifted(turn):
        # Asked to act, ran tools, but never landed the mutation. Ending on a
        # question is a hand-back (the model punted instead of acting);
        # ending on a statement is silent drift (bake-off T2: searched,
        # narrated the JSON, never added). Both mean the action didn't happen.
        return "needs-user" if _ends_on_question(turn.final_text) else "failed"
    if _ends_on_question(turn.final_text) and not _has_successful_call(turn.tools):
        return "needs-user"
    return "done"


def _action_drifted(turn: TurnRecord) -> bool:
    """The user asked to mutate state, the turn ran at least one tool, yet no
    successful mutating call landed. A pure read/search can't satisfy an action
    request. Turns with no tool calls are left alone — the search-and-narrate
    drift this targets always shows tool activity, and a no-tool reply is too
    ambiguous to fail deterministically."""
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
