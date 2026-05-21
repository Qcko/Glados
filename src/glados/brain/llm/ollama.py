"""Ollama adapter implementing the `LLM` Protocol.

Streams NDJSON from `/api/chat`. Translates between our `LLMMessage` /
`LLMToolCall` types and Ollama's wire format. Tool names are sanitised over
the wire (`server.name` → `server__name`) and a reverse map restores them
on the way back; the `__` separator is reserved (server/tool names with
`__` are rejected, keeping the sanitisation injective).

Untrusted-content wrapping (ARCHITECTURE §7) lives in the Organizer, not
here: a `ToolSpec(untrusted=True)` causes the Organizer to wrap the
result in `<external>...</external>` before it reaches this adapter.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

import httpx

from ...core.adapters import LLMEvent, LLMMessage, LLMText, LLMToolCall, ToolSpec


class OllamaLLM:
    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.2,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._transport = transport
        # Lazily constructed on first `chat()` so we bind to the running
        # event loop rather than whichever loop happened to be current at
        # build time. Reused across calls — httpx keeps a connection pool
        # internally, so streaming requests reuse keep-alive sockets to
        # Ollama instead of doing TCP+HTTP setup per turn.
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        # Intentionally sync: the None-check and assignment cannot interleave
        # without an `await` between them, so concurrent first-`chat()` calls
        # share one client. Do NOT add `await` here — that would open a
        # double-construct window and leak a socket pool.
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
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
            "options": {"temperature": self._temperature},
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
                for event in self._events_from_chunk(chunk, name_map):
                    yield event

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
