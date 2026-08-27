"""Recover tool calls that arrive as assistant TEXT rather than as structure.

Mistral's Ministral 3 emits `[TOOL_CALLS]<name>[ARGS]<json>`. Ollama does not
recognise that shape, so it hands the call back as ordinary content and the
turn looks like a model that answered in prose. Without this, GLaDOS would
SPEAK the call instead of running it.

TRUST BOUNDARY (ARCHITECTURE section 7). Parsing dispatches out of free text
turns anything that reaches the spoken channel into a potential action, and
tool results carry third-party text under `<external>`. Three rules keep that
from becoming an injection path, and none of them may be relaxed for
convenience:

  1. Opt-in per model. Silence here means "never parse text as a call".
  2. The marker must START the reply. A call quoted mid-sentence -- which is
     what echoed external content looks like -- is left as text.
  3. The name must be in the tools offered for THIS turn. An unrecognised name
     is NOT dispatched; unlike the structured path, there is no "unknown
     server" fallback, because inventing a dispatch from prose is exactly the
     move an attacker wants.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

CALL = "[TOOL_CALLS]"
THINK_OPEN = "[THINK]"
ARGS = "[ARGS]"
# Reasoning variants think first. An unclosed block must not swallow the call
# that follows it, so the run also ends at the next marker.
THINK = re.compile(r"^\s*\[THINK\].*?(\[/THINK\]|(?=\[TOOL_CALLS\])|$)", re.DOTALL)
MAX_CALLS_PER_TURN = 3

FORMAT_MISTRAL_V13 = "mistral_v13"
SUPPORTED_FORMATS = frozenset({FORMAT_MISTRAL_V13})


def parse_tool_text(
    raw: str, allowed: frozenset[str], *, fmt: str = FORMAT_MISTRAL_V13
) -> tuple[str, list[tuple[str, dict]], str]:
    """Split assistant output into (speakable text, calls, reasoning).

    `allowed` is the sanitised tool names offered this turn. Nothing from the
    marker onward is ever returned as speech: on any refusal the caller gets
    only the text that PRECEDED the marker, so a refused call is silence
    rather than markup read aloud.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown tool-text format: {fmt}")
    thought, body = _split_reasoning(raw)
    marker = body.find(CALL)
    if marker < 0:
        return body, [], thought
    before = body[:marker].strip()
    if before:
        # A call opens the reply; a marker with narration in front of it is
        # what an echo of <external> content looks like. Speak the narration,
        # dispatch nothing.
        log.warning("refusing tool text that does not open the reply")
        return before, [], thought
    calls, rest = _read_calls(body[marker:], allowed)
    if calls is None:
        return "", [], thought
    return _drop_any_marker(rest), calls, thought


def _split_reasoning(raw: str) -> tuple[str, str]:
    """Peel a leading [THINK] block off; it must never reach the speaker."""
    thought = THINK.match(raw)
    if not thought:
        return "", raw.lstrip()
    return thought.group(0), raw[thought.end() :].lstrip()


def _read_calls(
    text: str, allowed: frozenset[str]
) -> tuple[list[tuple[str, dict]] | None, str]:
    """Read consecutive calls off the front. None means "refuse the turn"."""
    calls: list[tuple[str, dict]] = []
    rest = text
    while rest.startswith(CALL) and len(calls) < MAX_CALLS_PER_TURN:
        name, sep, after = rest[len(CALL) :].partition(ARGS)
        if not sep:
            log.warning("refusing truncated tool text")
            return None, ""
        args, consumed = _json_prefix(after)
        name = name.strip()
        if name not in allowed:
            # Refuse rather than dispatch-and-let-the-registry-complain: the
            # structured path can afford that, prose cannot.
            log.warning("refusing tool text naming an unoffered tool: %s", name)
            return None, ""
        calls.append((name, args))
        rest = after[consumed:].lstrip()
    return calls, rest


def _drop_any_marker(text: str) -> str:
    """Whatever follows the last accepted call must not be spoken if it is
    itself a marker -- that is what hitting the per-turn cap looks like."""
    marker = text.find(CALL)
    return text if marker < 0 else text[:marker].strip()


def _json_prefix(text: str) -> tuple[dict, int]:
    """Read one JSON object off the FRONT of `text`, returning chars consumed.

    Splitting on the next marker instead would corrupt any argument value that
    contains one, so scan balanced braces and respect string escapes. Anything
    unreadable consumes to the end: a half-written call is not speakable text.
    """
    pad = len(text) - len(text.lstrip())
    text = text[pad:]
    if not text.startswith("{"):
        return {}, pad + len(text)
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text):
        if esc:
            esc = False
        elif in_str and ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str and ch == "{":
            depth += 1
        elif not in_str and ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[: i + 1]), pad + i + 1
                except json.JSONDecodeError:
                    return {}, pad + i + 1
    return {}, pad + len(text)


def could_start_call(text: str) -> bool:
    """True while `text` may still grow into a call marker.

    Streaming: the adapter must not speak a partial `[TOOL_` before knowing
    whether it is a call, but must not stall a normal reply either. A reasoning
    prelude also holds, because the call it may be introducing has not arrived.
    """
    stripped = text.lstrip()
    if not stripped:
        return True
    return _is_prefix_of(stripped, THINK_OPEN) or _is_prefix_of(stripped, CALL)


def _is_prefix_of(text: str, marker: str) -> bool:
    """True if `text` starts with `marker`, or could still grow into it."""
    return text.startswith(marker) or marker.startswith(text)
