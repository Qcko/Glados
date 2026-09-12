"""The spoken form of a confirmation question (DESIGN-voice-confirm.md).

Deterministic and harness-authored: the model never phrases what the user is
asked to approve, because the grant decision must not depend on a small
local model following a prompt. Fixed-typed arguments are spoken before
free text, and free text is wrapped as a quoted unit, so a value cannot mimic
a following argument ("tomatoes, quantity one, quantity 40"). A question that
would need clipping is not spoken at all -- the dialog blocks Allow until a
clipped value is expanded, and a spoken twin has no expand.
"""

from __future__ import annotations

import json

MAX_ARGS = 8
MAX_VALUE_CHARS = 60
MAX_BODY_CHARS = 240

_PREFIX = "GLaDOS needs a yes: "
# Ends on a word outside the answer lexicon: a room whose mic is not gated
# (the desk) can transcribe the tail of the question, and "...or no?" split
# by the VAD would read as a self-deny.
_SUFFIX = " -- shall I go ahead?"


def tts_safe(raw: str, limit: int) -> str:
    """What is safe to hand to TTS: printable characters only, whitespace
    collapsed, bounded. `_strip_markdown_for_tts` downstream is a prosody
    fix, not a sanitiser."""
    printable = "".join(ch for ch in raw if ch.isprintable() or ch.isspace())
    return " ".join(printable.split())[:limit].strip()


def render_confirm_question(tool_qualified: str, args: dict) -> str | None:
    """The question to speak for `tool_qualified(args)`, or None when it
    would not fit the spoken limits and must go to the screen only."""
    if len(args) > MAX_ARGS:
        return None
    rendered = _render_args(args)
    tool_words = _tool_words(tool_qualified)
    if rendered is None or tool_words is None:
        return None
    body = ", ".join([tool_words, *rendered])
    if len(body) > MAX_BODY_CHARS:
        return None
    return f"{_PREFIX}{body}{_SUFFIX}"


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
        return "yes" if value else "no"
    return str(value)


def _text_words(value: object) -> str | None:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(raw) > MAX_VALUE_CHARS:
        return None
    spoken = tts_safe(raw, MAX_VALUE_CHARS)
    return spoken if spoken else "empty"
