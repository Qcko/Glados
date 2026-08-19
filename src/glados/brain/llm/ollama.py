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

from ...core.adapters import LLMEvent, LLMMessage, LLMText, LLMToolCall, ToolSpec

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
        model: str = "qwen3:4b",
        temperature: float = 0.2,
        timeout: float = 60.0,
        keep_alive: str = "-1",
        # None means "the caller did not say" -- the key is omitted and Ollama
        # applies its own default. configs/glados.toml is the authoritative
        # source of the real values (LLMConfig), so these stay None rather than
        # duplicating numbers that would then drift out of step with it.
        num_ctx: int | None = None,
        num_predict: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._num_ctx = num_ctx
        self._num_predict = num_predict
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
        if name_map:
            payload["tools"] = [self._to_ollama_tool(k, v) for k, v in name_map.items()]

        client = self._ensure_client()
        async with client.stream(
            "POST", f"{self._host}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    self._log_usage(chunk)
                for event in self._events_from_chunk(chunk, name_map):
                    yield event

    def _build_options(self) -> dict:
        options: dict = {"temperature": self._temperature}
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        return options

    def _log_usage(self, chunk: dict) -> None:
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
        if done_reason == "length":
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
        return OllamaLLM._sanitise_pair(spec.server, spec.name)

    @staticmethod
    def _sanitise_pair(server: str, name: str) -> str:
        if "__" in server or "__" in name:
            raise ValueError(
                f"server/tool names must not contain '__' (reserved separator): {server}.{name}"
            )
        return f"{server}__{name}"

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
                            "name": OllamaLLM._sanitise_pair(tc.server, tc.name),
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
