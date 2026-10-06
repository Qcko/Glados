"""What a cart-write turn actually did, read from the cart itself.

DESIGN-cart-verify.md: on a cart-write turn whose real change could be read,
GLaDOS speaks a line built here from the BEFORE -> AFTER diff, not the model's
reply. Pure functions only; `CartVerifier` owns the reads and the cache.

The cart read is the shop's output and untrusted (ARCHITECTURE section 7), so
the parse is strict and all-or-nothing, keeps no product names, and the line
is built only from bounded integers and the calls' own subject words.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from glados.core.turn_outcome import _plain_subject

MAX_RAW_BYTES = 64 * 1024
_MAX_LINES = 100
_MAX_UNITS = 999
_MAX_PACK = 99
_PRODUCT_ID_RE = re.compile(r"[0-9]{1,12}", re.ASCII)
_MAX_NAMED = 3
_MAX_QUESTION_CHARS = 160
_SENTENCE_RE = re.compile(r"(?:\d\.\d|[^.!?])+(?:[.!?]+|$)")

NOTHING_CHANGED = "Nothing in your cart changed."

# GLaDOS-voiced tails for the cart line, behind `[cart_verify] persona`. Dry,
# not insulting -- one plays on every cart write. Harness-authored and free of
# numbers, product words and claims, so a tail can never make a line untrue.
_QUIPS = {
    "added": (
        "Noted. For science.",
        "The cart obliges.",
        "Another entry in the record.",
    ),
    "removed": (
        "Restraint. How refreshing.",
        "Gone. The cart is lighter for it.",
        "Duly subtracted.",
    ),
    "mixed": (
        "Rearranged.",
        "The cart adapts.",
    ),
}


@dataclass(frozen=True)
class CartLine:
    units: int
    pack_of: int | None = None


@dataclass(frozen=True)
class CartSnapshot:
    lines: Mapping[str, CartLine]

    @classmethod
    def parse(cls, raw: str) -> CartSnapshot | None:
        """The cart as productId -> units, or None if anything about the
        payload is not exactly the contract. Never a partial snapshot."""
        if not isinstance(raw, str):
            return None
        try:
            if len(raw.encode("utf-8")) > MAX_RAW_BYTES:
                return None
            payload = json.loads(raw, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            return None
        return _snapshot_from(payload)

    @classmethod
    def from_content(cls, content: object) -> CartSnapshot | None:
        """The same contract for a tool result the MCP client already decoded:
        re-encoded strictly so the byte cap and the NaN refusal still hold."""
        try:
            raw = json.dumps(content, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return None
        return cls.parse(raw)


@dataclass(frozen=True)
class LineChange:
    product_id: str
    before: int
    after: int
    pack_of: int | None

    @property
    def delta(self) -> int:
        return self.after - self.before


def diff(before: CartSnapshot, after: CartSnapshot) -> tuple[LineChange, ...]:
    """Every productId whose units moved, in id order. A line absent on one
    side counts as 0 units there."""
    changes = []
    for product_id in sorted(set(before.lines) | set(after.lines)):
        was = before.lines.get(product_id)
        now = after.lines.get(product_id)
        units_before = was.units if was else 0
        units_after = now.units if now else 0
        if units_before != units_after:
            pack_of = (now or was).pack_of
            changes.append(LineChange(product_id, units_before, units_after, pack_of))
    return tuple(changes)


def cart_line(
    changes: tuple[LineChange, ...],
    words: Mapping[str, str],
    question: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> str | None:
    """The sentence GLaDOS speaks for this turn's cart change.

    `words` attributes a productId to the subject word of the call that
    touched it; lines without one are described by direction only.
    `overrides` swaps a subject's sentence for one the caller built from a
    typed result (a volume add's litres). A word that is not a plain subject
    is never spoken -- the model wrote it after reading shop text, and on prod
    it is often a full shop product name -- so its line is described by
    direction instead."""
    if not changes:
        return f"{NOTHING_CHANGED} {question}" if question else NOTHING_CHANGED
    subjects = {pid: _plain_subject(word) for pid, word in words.items()}
    groups, unnamed = _group_by_subject(changes, subjects)
    named = list(groups.items())[:_MAX_NAMED]
    for _, group in list(groups.items())[_MAX_NAMED:]:
        unnamed.extend(group)
    sentences = [
        (overrides or {}).get(word) or _group_sentence(word, group)
        for word, group in named
    ]
    sentences += _unnamed_sentences(unnamed, other=bool(named))
    return " ".join(sentences)


def with_quip(line: str, changes: tuple[LineChange, ...], turn_no: int) -> str:
    """The cart line with a persona tail, rotated by `turn_no`. A line that
    ends on the model's kept question is returned as is: the question must
    stay last, or the user is not sure what to answer. Nothing changed gets
    no tail either: the line already says so, and a wry aside there can
    sound like a joke about a failed request."""
    kind = _change_kind(changes)
    if kind is None or line.rstrip().endswith("?"):
        return line
    quips = _QUIPS[kind]
    return f"{line} {quips[turn_no % len(quips)]}"


def _change_kind(changes: tuple[LineChange, ...]) -> str | None:
    raised = any(c.delta > 0 for c in changes)
    lowered = any(c.delta < 0 for c in changes)
    if raised and lowered:
        return "mixed"
    if raised:
        return "added"
    return "removed" if lowered else None


def trailing_question(reply: str) -> str | None:
    """The reply's last sentence when it is a question -- the one part of a
    no-change reply worth keeping ("Which milk did you mean?"). Capped in
    length: it is model text, spoken in GLaDOS's voice."""
    sentences = _SENTENCE_RE.findall(reply.strip())
    if not sentences:
        return None
    last = sentences[-1].strip()
    if "?" not in last[-3:] or len(last) > _MAX_QUESTION_CHARS:
        return None
    return last


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-finite number {name} in cart read")


def _snapshot_from(payload: object) -> CartSnapshot | None:
    if not isinstance(payload, dict):
        return None
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list) or len(raw_lines) > _MAX_LINES:
        return None
    lines: dict[str, CartLine] = {}
    for raw_line in raw_lines:
        parsed = _line_from(raw_line)
        if parsed is None or parsed[0] in lines:
            return None
        lines[parsed[0]] = parsed[1]
    return CartSnapshot(lines=lines)


def _line_from(raw_line: object) -> tuple[str, CartLine] | None:
    if not isinstance(raw_line, dict):
        return None
    product_id = raw_line.get("productId")
    units = raw_line.get("quantity")
    pack_of = raw_line.get("packOf")
    if not isinstance(product_id, str) or not _PRODUCT_ID_RE.fullmatch(product_id):
        return None
    if not _bounded_int(units, 0, _MAX_UNITS):
        return None
    if pack_of is not None and not _bounded_int(pack_of, 1, _MAX_PACK):
        return None
    return product_id, CartLine(units=units, pack_of=pack_of)


def _bounded_int(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _group_by_subject(
    changes: tuple[LineChange, ...], subjects: Mapping[str, str | None]
) -> tuple[dict[str, list[LineChange]], list[LineChange]]:
    groups: dict[str, list[LineChange]] = {}
    unnamed: list[LineChange] = []
    for change in changes:
        subject = subjects.get(change.product_id)
        if subject is None:
            unnamed.append(change)
        else:
            groups.setdefault(subject, []).append(change)
    return groups, unnamed


def _group_sentence(word: str, group: list[LineChange]) -> str:
    added = sum(c.delta for c in group if c.delta > 0)
    taken = -sum(c.delta for c in group if c.delta < 0)
    if added and taken:
        return f"Changed the {word}."
    if added:
        return f"Added {_count(added, group)} {word}."
    if not any(c.after for c in group):
        return f"Took the {word} out."
    return f"Took {_count(taken, group)} {word} off."


def _count(units: int, group: list[LineChange]) -> str:
    """Units, with the pack count beside them only when one line is involved
    and its pack size divides exactly -- never a rounded pack figure."""
    pack_of = group[0].pack_of if len(group) == 1 else None
    if pack_of and pack_of > 1 and units % pack_of == 0:
        packs = units // pack_of
        noun = "pack" if packs == 1 else "packs"
        return f"{units} ({packs} {noun} of {pack_of})"
    return str(units)


def _unnamed_sentences(changes: list[LineChange], *, other: bool) -> list[str]:
    """Lines no call's word names, by direction and count only."""
    gone = [c for c in changes if c.before and not c.after]
    lowered = [c for c in changes if c.after and c.delta < 0]
    raised = [c for c in changes if c.delta > 0]
    sentences = []
    if raised:
        sentences.append(f"Added {_units_of(raised, other)} to your cart.")
    if lowered:
        units_off = -sum(c.delta for c in lowered)
        sentences.append(f"Took {units_off} off {_items(len(lowered), other)}.")
    if gone:
        sentences.append(f"Took {_units_of(gone, other)} out of your cart.")
    return sentences


def _units_of(group: list[LineChange], other: bool) -> str:
    """How many units moved on how many lines. The line count alone said
    "Added one item" for one line going 1 -> 3 (prod bake-off T10,
    06-10-2026), so units are spoken whenever they differ from it."""
    units = sum(abs(c.delta) for c in group)
    items = _items(len(group), other)
    if units == len(group):
        return items
    if len(group) == 1:
        return f"{_count(units, group)} of {items}"
    return f"{units} units of {items}"


def _items(count: int, other: bool) -> str:
    prefix = "other " if other else ""
    if count == 1:
        return f"one {prefix}item"
    return f"{count} {prefix}items"
