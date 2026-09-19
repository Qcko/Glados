"""Names for the ids a tool call carries (DESIGN-voice-confirm.md).

A model that has just seen a cart or a search picks the id-keyed mutator,
and the confirm question then reads "remove product 1 0 0, 8 0 6, 8 9 3" --
digits nobody at the desk can check. This remembers, per room and per
server, every id-to-name pair the server's own results have shown, so the
question can say what the id names. The name is the server's word, exactly
as its by-name tools already speak it; the raw id stays on the dialog.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

# Argument keys whose value is an id worth naming. A result names an id
# either as a sibling `name` in a record, or inline as "Name (productId=N)".
ID_KEYS = frozenset({"productId"})
NAME_KEY = "name"
MAX_NAME_CHARS = 60
REMEMBERED_PER_SCOPE = 512

_INLINE_ID_RE = re.compile(r"\b(\w+)=(\d+)")
# Where a name can start, scanning back from its id: a line, a
# list separator, a "top:" style label, or a report verb with its count
# ("Added 2 x "). The count is not a lead on its own: "Volvic Water 6 x
# 1.5l" carries one inside the name (seen live, spoken as "remove product
# 1.5l"). Any other run-up ("Could not add NAME (", "Found NAME (") is not
# a name this can trust, and the id stays digits.
_NAME_LEAD_RE = re.compile(
    r"(?:\n|;\s*|--\s*|\w+:\s*"
    r"|\b(?:added|removed|changed|updated)\s+(?:\d+\s*x\s+)?)",
    re.IGNORECASE,
)
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class Named:
    key: str
    id: str
    name: str


@dataclass(frozen=True)
class _Known:
    name: str
    from_record: bool


class ProductNames:
    """Bounded, scope-keyed memory of what each id was last called."""

    def __init__(self) -> None:
        self._known: dict[tuple[str, str], OrderedDict[tuple[str, str], _Known]] = {}

    def learn(self, room_id: str, server: str, content: object) -> int:
        """Remember every id-to-name pair `content` shows; returns how many.
        A name read out of a record outranks one guessed from prose: the
        prose form is a heuristic, and a wrong guess must not replace the
        server's own field."""
        pairs = list(_pairs_in(content))
        if not pairs:
            return 0
        scope = self._known.setdefault((room_id, server), OrderedDict())
        for key, id_, name, from_record in pairs:
            held = scope.get((key, id_))
            if held is not None and held.from_record and not from_record:
                continue
            scope[(key, id_)] = _Known(name, from_record)
            scope.move_to_end((key, id_))
            while len(scope) > REMEMBERED_PER_SCOPE:
                scope.popitem(last=False)
        return len(pairs)

    def resolve(self, room_id: str, server: str, args: dict) -> tuple[dict, list[Named]]:
        """`args` with every known id replaced by its name, and what was
        replaced. An unknown id is left as it is."""
        scope = self._known.get((room_id, server))
        if not scope:
            return args, []
        named: list[Named] = []
        resolved = dict(args)
        for key, value in args.items():
            if key not in ID_KEYS or isinstance(value, bool):
                continue
            if not isinstance(value, (str, int)):
                continue
            held = scope.get((key, str(value)))
            if held is None:
                continue
            resolved[key] = held.name
            named.append(Named(key, str(value), held.name))
        return resolved, named


def _pairs_in(content: object):
    if isinstance(content, dict):
        yield from _pairs_in_record(content)
        for value in content.values():
            yield from _pairs_in(value)
    elif isinstance(content, list):
        for item in content:
            yield from _pairs_in(item)
    elif isinstance(content, str):
        yield from _pairs_in_text(content)


def _pairs_in_record(record: dict):
    name = record.get(NAME_KEY)
    if not isinstance(name, str):
        return
    for key in ID_KEYS:
        id_ = record.get(key)
        if isinstance(id_, (str, int)) and not isinstance(id_, bool):
            cleaned = _clean(name)
            if cleaned:
                yield key, str(id_), cleaned, True


def _pairs_in_text(text: str):
    for match in _INLINE_ID_RE.finditer(text):
        key, id_ = match.groups()
        if key not in ID_KEYS:
            continue
        name = _name_before(text[: match.start()])
        if name:
            yield key, id_, name, False


def _name_before(prefix: str) -> str | None:
    """The name that precedes an id: "Added 2 x Irish Milk 3L (productId="
    -> "Irish Milk 3L", "(top: Whole Milk 2L, productId=" -> "Whole Milk
    2L". Only an id in brackets or after a comma has a name in front of it
    ("waiting for: removing productId=1" has none), and only a name behind a
    known lead is taken; anything else is None and the id stays digits."""
    tail = prefix.rstrip()
    if not tail.endswith(("(", ",")):
        return None
    window = tail[:-1].rstrip()[-(MAX_NAME_CHARS + 20) :]
    start = _name_start(window)
    if start is None:
        return None
    name = _clean(window[start:])
    return name if _HAS_LETTER_RE.search(name) else None


def _name_start(window: str) -> int | None:
    """Where the name begins, or None when no lead precedes it -- the start
    of the text is not one: "Could not add NAME (productId=" opens with a
    run-up this cannot tell from a name, and digits beat a wrong name."""
    start = None
    for lead in _NAME_LEAD_RE.finditer(window):
        start = lead.end()
    return start


def _clean(name: str) -> str:
    printable = "".join(ch for ch in name if ch.isprintable())
    return " ".join(printable.split())[:MAX_NAME_CHARS].strip()
