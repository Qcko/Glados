"""Ollama adapter implementing the `LLM` Protocol.

Streams NDJSON from `/api/chat`. Translates between our `LLMMessage` /
`LLMToolCall` types and Ollama's wire format. Tool names are sanitised over
the wire (`server.name` -> `server__name`) and a reverse map restores them
on the way back; the `__` separator is reserved (server/tool names with
`__` are rejected, keeping the sanitisation injective).

Untrusted-content wrapping (ARCHITECTURE section 7) lives in the Organizer, not
here: a `ToolSpec(untrusted=True)` causes the Organizer to wrap the
result in `<external>...</external>` before it reaches this adapter.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

import httpx

from .tool_text import could_start_call, parse_tool_text
from .wire_names import UNKNOWN_SERVER, sanitise, sanitise_pair
from ...core.adapters import (
    LLMEvent,
    LLMMessage,
    LLMText,
    LLMThinking,
    LLMToolCall,
    ToolSpec,
)

log = logging.getLogger(__name__)

# Fraction of num_ctx the assembled prompt may reach before each turn warns.
# Ollama truncates from the front without reporting it, so the only signal that
# the system prompt is about to be evicted is the prompt size itself.
_CONTEXT_PRESSURE_RATIO = 0.8


class OllamaLLM:
    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        model: str = "ministral3:8b-instruct",
        temperature: float = 0.0,
        timeout: float = 60.0,
        keep_alive: str = "-1",
        # None means "the caller did not say" -- the key is omitted and Ollama
        # applies its own default. configs/glados.toml is the authoritative
        # source of the real values (LLMConfig), so these stay None rather than
        # duplicating numbers that would then drift out of step with it.
        num_ctx: int | None = None,
        num_predict: int | None = None,
        # Same None-means-omit contract, and the one option here whose absence
        # is NOT neutral: Ollama fills in 1.1 and llama.cpp fills in 1.0, so an
        # omitted value belongs to the runtime rather than to us. Production
        # passes it from LLMConfig, which is where the measured rationale lives.
        repeat_penalty: float | None = None,
        think: bool | None = None,
        # Wire format for tool calls the server hands back as TEXT (currently
        # only "mistral_v13"). STRICTLY PER-MODEL, like `think` above: None
        # means the text channel is never treated as a dispatch, which is the
        # right answer for every model whose calls Ollama already parses.
        #
        # Defaulted to match the default `model` above, since that one needs
        # it -- an incoherent pair here dispatches nothing and SPEAKS the raw
        # marker. Production passes both explicitly from LLMConfig, which
        # refuses a mismatched pair outright.
        text_tool_format: str | None = "mistral_v13",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._repeat_penalty = repeat_penalty
        self._think = think
        self._text_tool_format = text_tool_format
        # Context pressure is a standing condition, not an event: on a workload
        # whose tool block is genuinely large it would be true on EVERY turn,
        # and a warning that fires every turn stops being read -- which is how
        # the missing-num_ctx bug survived this long. Warn on the first crossing
        # only; the per-turn numbers stay available at info level.
        self._context_pressure_warned = False
        # Ollama's /api/chat `keep_alive`: a number is SECONDS (with -1 the
        # "resident forever" sentinel), a string must be a unit duration
        # ("30m", "1h"). A bare-number string like "-1" is rejected (400), so
        # coerce a numeric config value to int and leave unit strings as-is.
        # Resident-forever keeps the model from evicting mid-session (a re-cold
        # model drifts language / skips tools on the next free-form turn).
        self._keep_alive: int | str = self._coerce_keep_alive(keep_alive)
        self._transport = transport
        # Lazily constructed on first `chat()` so we bind to the running
        # event loop rather than whichever loop happened to be current at
        # build time. Reused across calls -- httpx keeps a connection pool
        # internally, so streaming requests reuse keep-alive sockets to
        # Ollama instead of doing TCP+HTTP setup per turn.
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _coerce_keep_alive(value: str) -> int | str:
        # A bare number ("-1", "300") rides as int seconds (-1 = resident
        # forever); a unit duration ("30m") stays a string. bool is an int
        # subclass and int(True)==1 silently, so reject it explicitly; a blank
        # value falls through to Ollama's own default rather than int("")=error.
        if isinstance(value, bool) or not str(value).strip():
            return value
        try:
            return int(value)
        except ValueError:
            return value

    def _ensure_client(self) -> httpx.AsyncClient:
        # Intentionally sync: the None-check and assignment cannot interleave
        # without an `await` between them, so concurrent first-`chat()` calls
        # share one client. Do NOT add `await` here -- that would open a
        # double-construct window and leak a socket pool.
        if self._client is None:
            # Streaming LLM: bound connect/write/pool to the configured
            # timeout, but use a long per-chunk read timeout. Ollama can
            # stall many seconds between tokens on a cold model or long
            # prompt; the previous default (60s shared with dispatch)
            # killed two turns mid-generation during the 2026-05-26 demo.
            # 600s is the safety net so a fully-wedged Ollama still
            # terminates instead of hanging the room queue forever; the
            # MCPRegistry dispatch timeout does NOT wrap LLM streams.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._timeout,
                    read=600.0,
                    write=self._timeout,
                    pool=self._timeout,
                ),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def price_prompt(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> int | None:
        """Exactly how many tokens this prompt costs, per the model itself.

        `num_predict=0` asks the server to evaluate the prompt and generate
        nothing, and the reply still carries `prompt_eval_count`. That is the
        real tokenizer through the real code path -- no bundled `tokenizer.json`
        to drift out of step with a swapped model, and no character estimate
        (measured 15% LOW, which is the unsafe direction for a budget).

        The catch, and why this prices boot-time checks rather than every turn:
        an over-long prompt is truncated here too, so the answer saturates at
        `num_ctx` and says THAT you overflowed without saying by how much. It
        can verify a prompt fits; it cannot compute how much to shed.

        Returns None if the daemon cannot be reached or answers without a
        count -- the caller decides whether an unknown size is fatal, since a
        boot check and a per-turn check disagree about that.
        """
        payload = {
            "model": self._model,
            "messages": [self._to_ollama_msg(m) for m in messages],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {**self._build_options(), "num_predict": 0},
        }
        if tools:
            name_map = {self._sanitise(t): t for t in tools}
            payload["tools"] = [
                self._to_ollama_tool(k, v) for k, v in name_map.items()
            ]
        try:
            response = await self._ensure_client().post(
                f"{self._host}/api/chat", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError:
            log.warning("could not price the prompt against %s", self._model)
            return None
        return response.json().get("prompt_eval_count")

    async def chat(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        name_map = {self._sanitise(t): t for t in tools}
        payload = {
            "model": self._model,
            "messages": [self._to_ollama_msg(m) for m in messages],
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": self._build_options(),
        }
        if self._think is not None:
            payload["think"] = self._think
        if name_map:
            payload["tools"] = [self._to_ollama_tool(k, v) for k, v in name_map.items()]

        client = self._ensure_client()
        async with client.stream(
            "POST", f"{self._host}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            # Whether the model ever produced something SPEAKABLE. Reasoning
            # tokens don't count: they bill against num_predict but never reach
            # the user, so a turn can burn the whole budget and still owe a
            # reply. `_log_usage` needs this to tell "cut mid-sentence" from
            # "cut before it said anything".
            produced_speakable = False
            # Text withheld from the spoken channel while it might still turn
            # out to be a tool call. Empty unless `text_tool_format` is set.
            held: list[str] = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                # Events BEFORE the usage log: the final chunk may itself carry
                # the content, and judging "did it say anything" without
                # counting that chunk would report a silent turn that spoke.
                events = self._events_from_chunk(chunk, name_map)
                produced_speakable = produced_speakable or any(
                    isinstance(e, (LLMText, LLMToolCall)) for e in events
                )
                if chunk.get("done"):
                    self._log_usage(chunk, produced_speakable=produced_speakable)
                for event in self._filtered(events, held):
                    yield event
            for event in self._drain(name_map, held):
                yield event
        if held:
            # Only reachable if the stream died or the consumer walked away
            # mid-turn. Nothing can be yielded now, so say so rather than let
            # a turn go silent with no explanation anywhere.
            log.warning(
                "dropped %d withheld chars: stream ended before drain model=%s",
                len("".join(held)),
                self._model,
            )

    def _filtered(self, events: list[LLMEvent], held: list[str]) -> list[LLMEvent]:
        """Hold back text that may be a tool call, pass everything else on.

        A model whose calls Ollama parses never reaches the holding branch, so
        streaming to the spoken channel is unchanged for it.
        """
        if self._text_tool_format is None:
            return events
        out: list[LLMEvent] = []
        for event in events:
            if not isinstance(event, LLMText):
                out.append(event)
                continue
            if held or could_start_call(event.text):
                held.append(event.text)
                if not could_start_call("".join(held)):
                    # Settled: it was ordinary speech all along. Release it and
                    # stop withholding for the rest of the turn.
                    out.append(LLMText(text="".join(held)))
                    held.clear()
                continue
            out.append(event)
        return out

    def _drain(
        self, name_map: dict[str, ToolSpec], held: list[str]
    ) -> list[LLMEvent]:
        """Turn whatever was withheld into calls, speech, or both."""
        if not held:
            return []
        raw = "".join(held)
        held.clear()
        spoken, calls, thought = parse_tool_text(
            raw, frozenset(name_map), fmt=self._text_tool_format
        )
        events: list[LLMEvent] = []
        if thought:
            # Reasoning belongs on the thinking channel whatever syntax it
            # arrived in. Speaking it is the failure LLMThinking exists to stop.
            events.append(LLMThinking(text=thought))
        for sanitised, args in calls:
            spec = name_map[sanitised]
            log.info(
                "ollama recovered tool call from text model=%s tool=%s.%s",
                self._model,
                spec.server,
                spec.name,
            )
            events.append(
                LLMToolCall(
                    call_id=uuid.uuid4().hex,
                    server=spec.server,
                    name=spec.name,
                    args=args,
                    from_text=True,
                )
            )
        if spoken.strip():
            events.append(LLMText(text=spoken))
        return events

    def _build_options(self) -> dict:
        options: dict = {"temperature": self._temperature}
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        if self._repeat_penalty is not None:
            options["repeat_penalty"] = self._repeat_penalty
        return options

    def _log_usage(self, chunk: dict, *, produced_speakable: bool) -> None:
        # The final chunk carries Ollama's own token accounting. Dropping it (as
        # this adapter did until 2026-08-17) is what made front-truncation
        # invisible: the prompt silently loses its head and nothing reports it.
        prompt_tokens = chunk.get("prompt_eval_count")
        reply_tokens = chunk.get("eval_count")
        done_reason = chunk.get("done_reason")
        if prompt_tokens is None:
            return
        log.info(
            "ollama chat done model=%s prompt_tokens=%s reply_tokens=%s "
            "num_ctx=%s done_reason=%s",
            self._model,
            prompt_tokens,
            reply_tokens,
            self._num_ctx,
            done_reason,
        )
        if done_reason == "length" and not produced_speakable:
            # Worse than a cut sentence: the budget went entirely on reasoning
            # and the turn owes the user a reply it never started. Says nothing,
            # yet every other signal reads as success -- so name it separately
            # rather than let it hide behind the mid-sentence wording below.
            log.warning(
                "model %s hit the num_predict=%s cap after %s tokens WITHOUT "
                "EMITTING ANY REPLY -- the whole budget went on reasoning, so "
                "this turn is silent, not merely truncated. Raise num_predict, "
                "or shrink the prompt this turn had to reason over.",
                self._model,
                self._num_predict,
                reply_tokens,
            )
        elif done_reason == "length":
            # The reply was cut at num_predict, so the spoken sentence just
            # stops. Without this the slice would trade one silent truncation
            # (the front of the prompt) for another (the tail of the reply).
            log.warning(
                "reply from model %s was truncated at the num_predict=%s cap "
                "after %s tokens -- the spoken reply is cut mid-sentence. Raise "
                "num_predict, or treat this as the repetition loop it bounds.",
                self._model,
                self._num_predict,
                reply_tokens,
            )
        if self._num_ctx is None:
            return
        if prompt_tokens > self._num_ctx * _CONTEXT_PRESSURE_RATIO:
            if self._context_pressure_warned:
                return
            self._context_pressure_warned = True
            log.warning(
                "assembled prompt is %s tokens against num_ctx=%s for model %s -- "
                "Ollama truncates from the FRONT, so the system prompt (and with "
                "it the <external> untrusted-content rule) is what gets evicted "
                "first. Raise num_ctx or shrink the tool list / history.",
                prompt_tokens,
                self._num_ctx,
                self._model,
            )

    @staticmethod
    def _sanitise(spec: ToolSpec) -> str:
        return sanitise(spec)

    @staticmethod
    def _sanitise_pair(server: str, name: str) -> str:
        return sanitise_pair(server, name)

    @staticmethod
    def _to_ollama_tool(sanitised_name: str, spec: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": sanitised_name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    @staticmethod
    def _to_ollama_msg(m: LLMMessage) -> dict:
        if m.role in ("system", "user"):
            return {"role": m.role, "content": m.content or ""}
        if m.role == "tool":
            out: dict = {"role": "tool", "content": m.content or ""}
            if m.tool_call_id:
                out["tool_call_id"] = m.tool_call_id
            return out
        if m.role == "assistant":
            out = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "function": {
                            # An `unknown`-server call already holds the raw
                            # wire name, which contains the reserved `__` by
                            # construction -- so sanitising it here raised and
                            # killed the turn on the NEXT pass. Reachable
                            # without an attacker: any hallucinated name of the
                            # shape `a__b` that was not offered this turn.
                            "name": (
                                tc.name
                                if tc.server == UNKNOWN_SERVER
                                else sanitise_pair(tc.server, tc.name)
                            ),
                            "arguments": tc.args,
                        },
                    }
                    for tc in m.tool_calls
                ]
            return out
        raise ValueError(f"unknown role: {m.role}")

    @staticmethod
    def _events_from_chunk(
        chunk: dict, name_map: dict[str, ToolSpec]
    ) -> list[LLMEvent]:
        events: list[LLMEvent] = []
        msg = chunk.get("message") or {}
        thinking = msg.get("thinking")
        if thinking:
            events.append(LLMThinking(text=thinking))
        content = msg.get("content")
        if content:
            events.append(LLMText(text=content))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            sanitised = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            spec = name_map.get(sanitised)
            if spec is None:
                # Surface hallucinated names as a tool_call with a sentinel
                # server. MCPRegistry returns "unknown tool", the LLM sees
                # the error, and the dropped name shows up in the trace.
                events.append(
                    LLMToolCall(
                        call_id=uuid.uuid4().hex,
                        server="unknown",
                        name=sanitised or "unnamed",
                        args=args,
                    )
                )
                continue
            events.append(
                LLMToolCall(
                    call_id=uuid.uuid4().hex,
                    server=spec.server,
                    name=spec.name,
                    args=args,
                )
            )
        return events
