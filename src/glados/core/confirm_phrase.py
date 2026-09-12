"""The spoken form of a confirmation question (DESIGN-voice-confirm.md).

Deterministic and harness-authored: the model never phrases what the user is
asked to approve, because the grant decision must not depend on a small
local model following a prompt. A tool may carry a `confirm_phrase` template
(servers.toml overlay) that reads as a sentence; without one, or whenever the
template cannot speak every argument the call carries, the generic form
speaks the tool name and each argument. Either way every argument present
in the call is heard, fixed-typed arguments come before free text (so a
value cannot mimic a following argument: "tomatoes, quantity one, quantity
40"), and a question that would need clipping is not spoken at all -- the
dialog blocks Allow until a clipped value is expanded, and a spoken twin has
no expand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .utterance import classify_confirm_answer

MAX_ARGS = 8
MAX_VALUE_CHARS = 60
MAX_BODY_CHARS = 240

# Names the one accepted token up front, far from the tail: a room whose mic
# is not gated (the desk) can transcribe the end of the question, and "yes"
# split off by the VAD there would read as a self-grant.
_PREFIX = "Say yes to: "
# Ends on a word outside the answer lexicon for the same reason.
_SUFFIX = " -- shall I go ahead?"

_GROUPED_DIGITS_MIN = 5
_ANSWER_FRAGMENT_RE = re.compile(r"""[,.;:!?"'(){}\[\]]+|\s-+\s""")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;!?])")

FIXED_SCHEMA_TYPES = frozenset({"integer", "number", "boolean"})


class PhraseError(ValueError):
    """A `confirm_phrase` template that does not parse. Raised at config
    load so a typo is a boot error, never a runtime fallback."""


@dataclass(frozen=True)
class Placeholder:
    name: str
    when_true: str | None = None
    when_false: str | None = None

    @property
    def is_switch(self) -> bool:
        return self.when_true is not None


@dataclass(frozen=True)
class Segment:
    """An optional stretch of the sentence, dropped whole when any
    placeholder inside names an argument absent from the call."""

    nodes: tuple[str | Placeholder, ...]


Node = str | Placeholder | Segment


@dataclass(frozen=True)
class Phrase:
    nodes: tuple[Node, ...]

    @property
    def placeholders(self) -> tuple[Placeholder, ...]:
        found: list[Placeholder] = []
        for node in self.nodes:
            if isinstance(node, Placeholder):
                found.append(node)
            elif isinstance(node, Segment):
                found.extend(n for n in node.nodes if isinstance(n, Placeholder))
        return tuple(found)

    def schema_problem(self, parameters: dict) -> str | None:
        """Why this template does not fit the tool's MCP input schema, or
        None. Checked once when the overlay meets the spec, so a renamed
        argument is a warning at server start rather than a silent fallback
        on every ask."""
        properties = parameters.get("properties") or {}
        text_seen = 0
        for placeholder in self.placeholders:
            if placeholder.name not in properties:
                return f"no argument named {placeholder.name!r}"
            kind = _schema_kind(properties[placeholder.name])
            if placeholder.is_switch and kind != "boolean":
                return f"{placeholder.name!r} is a switch but not boolean"
            if kind == "boolean" and not placeholder.is_switch:
                return f"{placeholder.name!r} is boolean and needs a switch"
            if kind != "text":
                if text_seen:
                    return f"fixed argument {placeholder.name!r} after free text"
                continue
            text_seen += 1
            if text_seen > 1:
                return f"more than one free-text argument ({placeholder.name!r})"
        return None


def _schema_kind(declared: object) -> str:
    """`"boolean"`, `"fixed"` or `"text"` for one property's schema. A type
    list is nullable-fixed only when every non-null member is fixed; any
    shape this does not understand (`anyOf`, `$ref`, a bare `true`) is free
    text, the ordering-strict side."""
    if not isinstance(declared, dict):
        return "text"
    kinds = declared.get("type")
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not isinstance(kinds, list) or not kinds:
        return "text"
    named = {k for k in kinds if k != "null"}
    if named == {"boolean"}:
        return "boolean"
    if named and named <= FIXED_SCHEMA_TYPES:
        return "fixed"
    return "text"


@lru_cache(maxsize=256)
def parse_phrase(template: str) -> Phrase:
    """The one parser for the template grammar: `{arg}` speaks a value,
    `{arg:A|B}` speaks A when a boolean is true and B when false, `[ ... ]`
    is dropped whole when an argument inside is absent. Cached, so the
    renderer works from the same tree the boot check accepted."""
    parser = _PhraseParser(template)
    return parser.parse()


class _PhraseParser:
    def __init__(self, template: str) -> None:
        self._text = template
        self._pos = 0
        self._names: set[str] = set()

    def parse(self) -> Phrase:
        nodes = self._nodes(closing=None)
        if not self._names:
            raise PhraseError("template names no argument")
        if _carries_an_answer(_literal_text(nodes)):
            raise PhraseError("template text reads as a yes or a no")
        return Phrase(tuple(nodes))

    def _nodes(self, closing: str | None) -> list[str | Placeholder | Segment]:
        nodes: list[str | Placeholder | Segment] = []
        literal: list[str] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == closing:
                self._pos += 1
                self._flush(nodes, literal)
                return nodes
            if ch == "{":
                self._flush(nodes, literal)
                nodes.append(self._placeholder())
            elif ch == "[":
                if closing is not None:
                    raise PhraseError("nested [ ] is not allowed")
                self._flush(nodes, literal)
                nodes.append(self._segment())
            elif ch in "]}":
                raise PhraseError(f"unmatched {ch!r}")
            else:
                literal.append(ch)
                self._pos += 1
        if closing is not None:
            raise PhraseError(f"missing {closing!r}")
        self._flush(nodes, literal)
        return nodes

    @staticmethod
    def _flush(nodes: list, literal: list[str]) -> None:
        if literal:
            nodes.append("".join(literal))
            literal.clear()

    def _segment(self) -> Segment:
        self._pos += 1
        inner = self._nodes(closing="]")
        if not any(isinstance(n, Placeholder) for n in inner):
            raise PhraseError("[ ] segment names no argument")
        return Segment(tuple(inner))

    def _placeholder(self) -> Placeholder:
        end = self._text.find("}", self._pos)
        if end < 0:
            raise PhraseError("missing '}'")
        body = self._text[self._pos + 1 : end]
        self._pos = end + 1
        name, sep, branches = body.partition(":")
        if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise PhraseError(f"bad placeholder name {name!r}")
        if name in self._names:
            raise PhraseError(f"argument {name!r} used twice")
        self._names.add(name)
        if not sep:
            return Placeholder(name)
        when_true, bar, when_false = branches.partition("|")
        if not bar or "|" in when_false:
            raise PhraseError(f"switch {name!r} needs exactly one '|'")
        return Placeholder(name, when_true, when_false)


def _literal_text(nodes: list) -> str:
    """Everything the author wrote that can be spoken: literals in place,
    each switch side as its own stretch."""
    pieces: list[str] = []
    for node in nodes:
        inner = node.nodes if isinstance(node, Segment) else (node,)
        for piece in inner:
            if isinstance(piece, str):
                pieces.append(piece)
            else:
                pieces.append(f", {piece.when_true or ''}, {piece.when_false or ''}, ")
    return "".join(pieces)


@dataclass(frozen=True)
class SpokenQuestion:
    """What `_ask_aloud` speaks, or why it does not. `fallback` names why a
    template was set aside for the generic form; `skipped` names why nothing
    is spoken at all (`text` is then None)."""

    text: str | None
    fallback: str | None = None
    fallback_keys: tuple[str, ...] = field(default_factory=tuple)
    skipped: str | None = None


def tts_safe(raw: str, limit: int) -> str:
    """What is safe to hand to TTS: printable characters only, whitespace
    collapsed, bounded. `_strip_markdown_for_tts` downstream is a prosody
    fix, not a sanitiser."""
    printable = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    return " ".join(printable.split())[:limit].strip()


def render_confirm_question(tool_qualified: str, args: dict) -> str | None:
    """The generic question for `tool_qualified(args)`, or None when it
    would not fit the spoken limits and must go to the screen only."""
    return compose_confirm_question(tool_qualified, args, None).text


def compose_confirm_question(
    tool_qualified: str, args: dict, template: str | None
) -> SpokenQuestion:
    if len(args) > MAX_ARGS:
        return SpokenQuestion(None, skipped="clipped")
    fallback: str | None = None
    keys: tuple[str, ...] = ()
    body: str | None = None
    if template is not None:
        body, fallback, keys = _render_phrase(parse_phrase(template), args)
    if body is None:
        body = _render_generic(tool_qualified, args)
    if body is None or len(body) > MAX_BODY_CHARS:
        return SpokenQuestion(None, fallback, keys, skipped="clipped")
    if _carries_an_answer(body):
        return SpokenQuestion(None, fallback, keys, skipped="answer_in_value")
    return SpokenQuestion(f"{_PREFIX}{body}{_SUFFIX}", fallback, keys)


def _carries_an_answer(spoken: str) -> bool:
    """A stretch of the question that classifies as yes or no could, split
    off by the VAD on an ungated mic, grant its own request. The body is
    checked whole, since a key the model chose, a value, or an author's
    literal can each carry it."""
    return any(
        classify_confirm_answer(window) is not None
        for fragment in _ANSWER_FRAGMENT_RE.split(spoken)
        for window in _word_windows(fragment)
    )


def _word_windows(fragment: str) -> list[str]:
    """Every run of words in a pause-delimited stretch: a VAD cut can land
    between any two words, so "add yes please to the cart" must count."""
    words = fragment.split()
    return [
        " ".join(words[start:end])
        for start in range(len(words))
        for end in range(start + 1, len(words) + 1)
    ]


def _render_phrase(
    phrase: Phrase, args: dict
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """The templated body, or (None, why, keys) when the template must be
    set aside for the generic form. Every check here is per call and
    fail-safe: the schema check at spec-merge time already rejected a
    template that could never fit."""
    spoken_names: list[str] = []
    parts: list[str] = []
    text_seen = 0
    for piece in _applicable_pieces(phrase, args):
        if isinstance(piece, str):
            parts.append(piece)
            continue
        words, problem = _placeholder_words(piece, args, text_seen)
        if words is None:
            return None, problem, (piece.name,)
        text_seen += not _is_fixed(args[piece.name])
        parts.append(words)
        spoken_names.append(piece.name)
    unheard = tuple(k for k in args if k not in spoken_names)
    if unheard:
        return None, "unmentioned", tuple(tts_safe(str(k), MAX_VALUE_CHARS) for k in unheard)
    return _tidy("".join(parts)), None, ()


def _applicable_pieces(phrase: Phrase, args: dict) -> list[str | Placeholder]:
    pieces: list[str | Placeholder] = []
    for node in phrase.nodes:
        if isinstance(node, Segment):
            if _segment_applies(node, args):
                pieces.extend(node.nodes)
        else:
            pieces.append(node)
    return pieces


def _segment_applies(segment: Segment, args: dict) -> bool:
    return all(n.name in args for n in segment.nodes if isinstance(n, Placeholder))


def _placeholder_words(
    placeholder: Placeholder, args: dict, text_seen: int
) -> tuple[str | None, str | None]:
    """What one placeholder speaks, or (None, why) when the call's value is
    not the shape the template was written for."""
    if placeholder.name not in args:
        return None, "missing"
    value = args[placeholder.name]
    if placeholder.is_switch:
        if not isinstance(value, bool):
            return None, "not_bool"
        return (placeholder.when_true if value else placeholder.when_false) or "", None
    if _is_fixed(value):
        if value is None or isinstance(value, bool):
            return None, "not_a_number"
        if text_seen:
            return None, "text_before_fixed"
        return _fixed_words(value), None
    if text_seen:
        return None, "multiple_text"
    spoken = _text_words(value)
    return (spoken, None) if spoken is not None else (None, "clipped_value")


def _tidy(body: str) -> str:
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", " ".join(body.split())).strip()


def _render_generic(tool_qualified: str, args: dict) -> str | None:
    rendered = _render_args(args)
    tool_words = _tool_words(tool_qualified)
    if rendered is None or tool_words is None:
        return None
    return ", ".join([tool_words, *rendered])


def _tool_words(name: str) -> str | None:
    """A dotted or snake name as words, or None when it would not fit --
    a name is never clipped either."""
    if len(name) > MAX_VALUE_CHARS:
        return None
    return tts_safe(name.replace(".", " ").replace("_", " "), MAX_VALUE_CHARS)


def _render_args(args: dict) -> list[str] | None:
    fixed: list[str] = []
    text: list[str] = []
    for key, value in args.items():
        spoken_key = _tool_words(str(key))
        if spoken_key is None:
            return None
        if _is_fixed(value):
            fixed.append(f"{spoken_key} {_fixed_words(value)}")
            continue
        spoken = _text_words(value)
        if spoken is None:
            return None
        text.append(f"{spoken_key}, quote, {spoken}, unquote")
    return fixed + text


def _is_fixed(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float))


def _fixed_words(value: object) -> str:
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and value < 0:
        return f"minus {_digits_spoken(str(-value))}"
    return _digits_spoken(str(value))


def _text_words(value: object) -> str | None:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(raw) > MAX_VALUE_CHARS:
        return None
    spoken = tts_safe(raw, MAX_VALUE_CHARS)
    return _digits_spoken(spoken) if spoken else "empty"


def _digits_spoken(spoken: str) -> str:
    """A long run of digits (a product id) read digit by digit in threes,
    "1 0 0, 8 0 6, 8 9 3": TTS would otherwise read it as a number in the
    hundreds of millions, and the listener could not check it either way."""
    if not spoken.isdigit() or len(spoken) < _GROUPED_DIGITS_MIN:
        return spoken
    groups = [spoken[i : i + 3] for i in range(0, len(spoken), 3)]
    return ", ".join(" ".join(g) for g in groups)
