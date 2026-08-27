"""llama.cpp adapter implementing the `LLM` Protocol.

Streams SSE from `llama-server`'s OpenAI-shaped `/v1/chat/completions`. Exists
to MEASURE, not to migrate: `DESIGN-llamacpp-runtime.md` defers the runtime
question, and the one dimension its evidence never covered is prose quality
under mistralai's canonical `chat_template.jinja` (`--jinja`) instead of the
497-byte template hand-ported into `configs/ministral3-8b-instruct.Modelfile`.
Running the 22-point suite on this adapter is what turns that deferral into a
finding.

Two differences from `OllamaLLM` are load-bearing rather than cosmetic:

**Tool calls arrive as STRUCTURE, so `tool_text.py` has no part here.** That
deletes the prose parser -- and with it rule 2, "the marker must START the
reply", which refused a call with narration in front of it precisely because a
mid-reply marker is what an echo of `<external>` content looks like. llama.cpp
accepts content-then-calls by design (measured: a probe asking for preamble
then a call returned both, with `finish_reason: "tool_calls"`). The parser has
not gone away either -- it moved inside llama-server, where GLaDOS cannot gate
it. What replaces rule 2 is the Organizer's untrusted-context gate, which keys
off the turn rather than off per-call provenance; `LLMToolCall.from_text` is
correctly False for everything here and must NOT be faked to trip the old arm.

**Arguments arrive as string FRAGMENTS across chunks**, keyed by `index`, so a
call is assembled rather than read. An assembly that does not parse is refused
(routed to the `unknown` sentinel), never dispatched with `args={}` -- on the
Ollama path that fallback is near-dead code because arguments arrive whole; on
a fragment stream, truncation is routine and the fallback would mean a MUTATING
tool dispatched with empty arguments.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

import httpx

from .tool_text import MAX_CALLS_PER_TURN
from .wire_names import UNKNOWN_SERVER, build_name_map, resolve, sanitise_pair
from ...core.adapters import (
    LLMEvent,
    LLMMessage,
    LLMText,
    LLMThinking,
    LLMToolCall,
    ToolSpec,
)

log = logging.getLogger(__name__)

_DONE = "[DONE]"
_CONTEXT_PRESSURE_RATIO = 0.8


class _PartialCall:
    """One in-flight tool call, accumulated across chunks.

    Keyed by the wire `index`, never by arrival order: a later chunk may carry
    only `index` plus an `arguments` fragment, and parallel calls interleave.
    """

    def __init__(self) -> None:
        self.call_id: str = ""
        self.name: str = ""
        self.args_text: list[str] = []

    def absorb(self, delta: dict) -> None:
        if delta.get("id"):
            self.call_id = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            self.name = fn["name"]
        if fn.get("arguments"):
            self.args_text.append(fn["arguments"])

    @property
    def dispatchable(self) -> bool:
        return bool(self.name)


class LlamaCppLLM:
    def __init__(
        self,
        *,
        host: str = "http://127.0.0.1:8080",
        model: str = "",
        temperature: float = 0.0,
        timeout: float = 60.0,
        # Sent explicitly, never inherited. llama-server's `--n-predict`
        # defaults to -1 (unbounded) where Ollama ran under `num_predict`, and
        # an unbounded reply is both the 2026-06-18 repetition loop unbounded
        # and a prose-length difference that would be misread as a
        # template-fidelity result. This is the `repeat_penalty` lesson
        # generalised: an omitted sampler belongs to the runtime, and a runtime
        # comparison cannot let the runtime pick.
        max_tokens: int | None = None,
        # Ollama fills in 1.1 when absent, llama.cpp fills in 1.0.
        repeat_penalty: float | None = None,
        # `top_p`/`top_k`/`min_p` are deliberately NOT plumbed. GLaDOS ships
        # `temperature = 0.0`, and greedy decoding takes the argmax logit --
        # which every TRUNCATING sampler retains by construction. They cannot
        # change a greedy pick, so pinning them would be ceremony.
        #
        # That reasoning covers truncating samplers only, and does not
        # generalise. XTC deliberately REMOVES top candidates, and DRY (like
        # repeat_penalty below) acts on the logits before the argmax. Both are
        # inert at llama-server's defaults but are launch-flag settable, so the
        # `llama-server` command line a measurement runs under is load-bearing
        # and must not set them.
        num_ctx: int | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._repeat_penalty = repeat_penalty
        self._num_ctx = num_ctx
        self._api_key = api_key
        self._transport = transport
        self._context_pressure_warned = False
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        # Intentionally sync, exactly as in OllamaLLM: the None-check and the
        # assignment cannot interleave without an `await` between them, so
        # concurrent first-`chat()` calls share one client. Do NOT add `await`
        # here -- that opens a double-construct window and leaks a socket pool.
        if self._client is None:
            # The read timeout is deliberately NOT the configured one. A
            # streaming server stalls many seconds between tokens on a cold
            # model or a long prompt, and a shared 60s killed two turns
            # mid-generation during the 2026-05-26 demo. Here it would be worse
            # than a bug: a truncated reply scores as bad PROSE, so the
            # measurement would report a model difference that is really a
            # timeout.
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._timeout,
                    read=600.0,
                    write=self._timeout,
                    pool=self._timeout,
                ),
                headers=headers,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        name_map = build_name_map(tools)
        payload: dict = {
            "model": self._model,
            "messages": [self._to_wire_msg(m) for m in messages],
            "stream": True,
            # Without this there is no usage record at all: no truncation
            # warning, no context-pressure warning, no token counts. A suite
            # whose replies were silently cut with nothing in the log saying so
            # produces a scorecard that reads as "worse prose".
            "stream_options": {"include_usage": True},
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._repeat_penalty is not None:
            payload["repeat_penalty"] = self._repeat_penalty
        if name_map:
            payload["tools"] = [
                self._to_wire_tool(k, v) for k, v in name_map.items()
            ]

        # Keyed by (choice index, tool-call index). The choice half matters
        # only if `n > 1` is ever sent, but without it index 0 of two choices
        # would merge: names overwrite and argument fragments concatenate into
        # JSON that cannot parse.
        partials: dict[tuple[int, int], _PartialCall] = {}
        finish_reason = ""
        # Reasoning does NOT count as speakable: it bills against max_tokens but
        # never reaches the user, so a turn can burn the whole budget and still
        # owe a reply. Counting it here would hide exactly the failure the
        # truncation warning exists to name.
        produced_speakable = False

        client = self._ensure_client()
        async with client.stream(
            "POST", f"{self._host}/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                chunk = self._parse_sse_line(line)
                if chunk is None:
                    continue
                self._log_usage(chunk)
                for choice in chunk.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    for event in self._events_from_delta(choice, partials):
                        produced_speakable = produced_speakable or isinstance(
                            event, LLMText
                        )
                        yield event

        drained = self._drain(partials, name_map, finish_reason)
        self._warn_if_truncated(
            finish_reason, produced_speakable=produced_speakable or bool(drained)
        )
        for event in drained:
            yield event

    @staticmethod
    def _parse_sse_line(line: str) -> dict | None:
        """SSE gives lines, not events: blank separators, `:` comments and the
        `[DONE]` sentinel all arrive here and none of them are JSON."""
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        body = line[len("data:") :].strip()
        if body == _DONE:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            log.warning("llamacpp: unparseable SSE data line, skipped")
            return None

    def _events_from_delta(
        self, choice: dict, partials: dict[tuple[int, int], _PartialCall]
    ) -> list[LLMEvent]:
        """Text streams straight through; tool calls only accumulate. A call is
        not an event until the stream says it is complete -- emitting a
        half-built one would dispatch a name with partial arguments."""
        delta = choice.get("delta") or {}
        events: list[LLMEvent] = []
        thinking = delta.get("reasoning_content")
        if thinking:
            events.append(LLMThinking(text=thinking))
        content = delta.get("content")
        if content:
            events.append(LLMText(text=content))
        choice_index = choice.get("index", 0)
        for tc in delta.get("tool_calls") or []:
            key = (choice_index, tc.get("index", 0))
            partials.setdefault(key, _PartialCall()).absorb(tc)
        return events

    def _drain(
        self,
        partials: dict[tuple[int, int], _PartialCall],
        name_map: dict[str, ToolSpec],
        finish_reason: str,
    ) -> list[LLMEvent]:
        """Emit the accumulated calls, in `index` order rather than the order
        their fragments happened to arrive."""
        events: list[LLMEvent] = []
        for index in sorted(partials):
            if len(events) >= MAX_CALLS_PER_TURN:
                # The same amplification cap `tool_text.py` applies. One echoed
                # injection should not fan out into an unbounded run of
                # mutating calls.
                log.warning(
                    "llamacpp: refusing tool calls beyond %d this turn model=%s",
                    MAX_CALLS_PER_TURN,
                    self._model,
                )
                break
            call = partials[index]
            if not call.dispatchable:
                log.warning(
                    "llamacpp: dropping tool call index=%s with no name "
                    "(stream ended mid-call, finish_reason=%s)",
                    index,
                    finish_reason,
                )
                continue
            args = self._parse_args(call)
            if args is None:
                # Refused, not defaulted. An assembly that does not parse is a
                # truncated or interleaved stream, and dispatching a MUTATING
                # tool with empty arguments is the worst available reading of
                # that. The sentinel makes the registry answer "unknown tool",
                # so the model sees an error and can retry.
                events.append(
                    LLMToolCall(
                        call_id=call.call_id or uuid.uuid4().hex,
                        server=UNKNOWN_SERVER,
                        name=call.name,
                        args={},
                    )
                )
                continue
            server, name = resolve(call.name, name_map)
            events.append(
                LLMToolCall(
                    call_id=call.call_id or uuid.uuid4().hex,
                    server=server,
                    name=name,
                    args=args,
                )
            )
        return events

    def _parse_args(self, call: _PartialCall) -> dict | None:
        raw = "".join(call.args_text).strip()
        if not raw:
            return {}
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(
                "llamacpp: tool call %r arguments did not parse (%d chars) -- "
                "refusing rather than dispatching with empty args",
                call.name,
                len(raw),
            )
            return None
        if not isinstance(args, dict):
            log.warning(
                "llamacpp: tool call %r arguments were %s, not an object",
                call.name,
                type(args).__name__,
            )
            return None
        return args

    def _warn_if_truncated(
        self, finish_reason: str, *, produced_speakable: bool
    ) -> None:
        """The reason `include_usage` is requested at all. A reply cut at
        `max_tokens` just stops mid-sentence, and in a prose-quality
        measurement that scores as bad writing with nothing in the log to say
        otherwise -- a wrong finding rather than a visible failure."""
        if finish_reason != "length":
            return
        if not produced_speakable:
            log.warning(
                "model %s hit the max_tokens=%s cap WITHOUT EMITTING ANY REPLY "
                "-- the whole budget went on reasoning, so this turn is silent, "
                "not merely truncated. Raise max_tokens, or shrink the prompt.",
                self._model,
                self._max_tokens,
            )
            return
        log.warning(
            "reply from model %s was truncated at the max_tokens=%s cap -- the "
            "spoken reply is cut mid-sentence. Raise max_tokens, or treat this "
            "as the repetition loop it bounds.",
            self._model,
            self._max_tokens,
        )

    def _log_usage(self, chunk: dict) -> None:
        usage = chunk.get("usage")
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens")
        reply_tokens = usage.get("completion_tokens")
        log.info(
            "llamacpp chat done model=%s prompt_tokens=%s reply_tokens=%s num_ctx=%s",
            self._model,
            prompt_tokens,
            reply_tokens,
            self._num_ctx,
        )
        if self._num_ctx is None or prompt_tokens is None:
            return
        if prompt_tokens > self._num_ctx * _CONTEXT_PRESSURE_RATIO:
            if self._context_pressure_warned:
                return
            self._context_pressure_warned = True
            log.warning(
                "assembled prompt is %s tokens against a CONFIGURED context of "
                "%s for model %s -- the real window is llama-server's launch "
                "-c, which this adapter cannot see, so treat this as the "
                "expectation rather than the server's truth. If it is right, "
                "the system prompt (and with it the <external> "
                "untrusted-content rule) is what gets evicted first.",
                prompt_tokens,
                self._num_ctx,
                self._model,
            )

    @staticmethod
    def _to_wire_tool(sanitised_name: str, spec: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": sanitised_name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }

    @staticmethod
    def _to_wire_msg(m: LLMMessage) -> dict:
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
                        "type": "function",
                        "function": {
                            # An `unknown`-server call is replayed under the
                            # name the model actually emitted. Sanitising the
                            # pair would raise on the `__` that name already
                            # contains, killing the turn on attacker-chosen
                            # bytes.
                            "name": (
                                tc.name
                                if tc.server == UNKNOWN_SERVER
                                else sanitise_pair(tc.server, tc.name)
                            ),
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in m.tool_calls
                ]
            return out
        raise ValueError(f"unknown role: {m.role}")
