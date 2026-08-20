"""Vendor-neutral truncation of a tool result before the LLM sees it.

A voice reply is read aloud end to end -- the listener cannot skim -- so a tool
that returns twenty items produces twenty items of speech. Capping the PAYLOAD
rather than prompting for brevity is the deterministic fix: the model cannot
name an item it never received.

Deliberately generic. This module knows nothing about any server's schema: it
slices a list and counts what it held back. Ranking and filtering belong to the
server that owns the data (ARCHITECTURE.md section 7 -- GLaDOS does not parse
another repo's payload in the path that handles hostile bytes).

Nothing here raises on bad input. An unexpected shape returns the payload
untouched, because the failure everyone can live with is the long list we have
today, not a turn that dies or a silent "you have no results"."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WITHHELD_KEY = "withheld_count"
SHOWN_KEY = "shown"


@dataclass(frozen=True)
class PayloadCap:
    """How many items of a tool result reach the model.

    `flex_to` exists because a cap that withholds one item is absurd out loud
    -- "here are five, want the other one?". Up to `flex_to` the whole list is
    spoken and no remainder is reported."""

    max_items: int
    flex_to: int | None = None
    items_key: str | None = None


@dataclass(frozen=True)
class CappedPayload:
    content: Any
    withheld: int
    # False when the payload held no list where one was configured. Distinct
    # from "the list was short enough to keep whole": one is schema drift worth
    # a log line, the other is the cap doing nothing because nothing was needed.
    recognised: bool

    @property
    def capped(self) -> bool:
        return self.withheld > 0


def cap_tool_payload(content: Any, cap: PayloadCap) -> CappedPayload:
    """Truncate `content` to `cap`, returning the payload the model should see
    and how many items were held back."""
    items = _items_of(content, cap.items_key)
    if items is None or not _cap_is_sane(cap):
        return CappedPayload(content, 0, recognised=items is not None)
    if len(items) <= _effective_limit(len(items), cap):
        return CappedPayload(content, 0, recognised=True)
    shown = items[: cap.max_items]
    withheld = len(items) - len(shown)
    return CappedPayload(
        _rebuilt(content, cap.items_key, shown, withheld), withheld, recognised=True
    )


def _items_of(content: Any, items_key: str | None) -> list[Any] | None:
    """The list this cap applies to, or None when the payload is not a shape we
    recognise -- which is the signal to leave it alone."""
    if items_key is None:
        return content if isinstance(content, list) else None
    if not isinstance(content, dict):
        return None
    nested = content.get(items_key)
    return nested if isinstance(nested, list) else None


def _cap_is_sane(cap: PayloadCap) -> bool:
    if cap.max_items < 1:
        return False
    return cap.flex_to is None or cap.flex_to >= cap.max_items


def _effective_limit(total: int, cap: PayloadCap) -> int:
    if cap.flex_to is not None and total <= cap.flex_to:
        return total
    return cap.max_items


def _rebuilt(content: Any, items_key: str | None, shown: list[Any], withheld: int) -> Any:
    """A new payload carrying the kept items and an HONEST count of the rest.
    The count is what stops the model inventing one when it offers the
    remainder, so it is derived from the list actually sliced -- never from a
    raw total the server reported."""
    if items_key is None:
        return {SHOWN_KEY: shown, WITHHELD_KEY: withheld}
    rebuilt = dict(content)
    rebuilt[items_key] = shown
    # Last write wins deliberately: a server field of the same name would be a
    # count of something else, and the model must not read it as this one.
    rebuilt[WITHHELD_KEY] = withheld
    return rebuilt
