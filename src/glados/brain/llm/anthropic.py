"""Anthropic Messages API adapter implementing the `LLM` Protocol.

The dormant cloud escape hatch for the v2.6 router. GLaDOS is all-local by
default (ARCHITECTURE §12); this adapter only runs when the specialist slot is
explicitly pointed at the cloud (`provider="anthropic"` + cloud opt-in + key).
It drives the *same* MCP tools as the local path — the registry doesn't care
which model emits the tool call — so the local side stays a pure dispatcher.

Streams Server-Sent Events from `/v1/messages`. Translates between our
`LLMMessage` / `LLMToolCall` types and Anthropic's content-block format. Tool
names are sanitised over the wire (`server.name` -> `server__name`, the `__`
separator reserved) and restored on the way back, matching the Ollama adapter.

Privacy: on this path tool *arguments and results* cross to Anthropic alongside
the transcript (ARCHITECTURE §9). Construction is gated on explicit opt-in +
an API key in the server wiring; this class assumes that gate already passed.
The endpoint is hardcoded to api.anthropic.com — there is no `base_url` knob,
which is what keeps the dormant code from being an arbitrary-destination exfil
path; adding one for a self-hosted endpoint is a separate guarded slice (§12).
Zero-retention is an account-level setting on Anthropic's side, not a per-call
body flag — there is nothing to send here for it.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

import httpx

from ...core.adapters import LLMEvent, LLMMessage, LLMText, LLMToolCall, ToolSpec

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = transport
        # Lazily built on first chat() so the pool binds to the running loop —
        # same rationale as OllamaLLM.
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
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
        system, api_messages = self._to_anthropic_messages(messages)
        payload: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": api_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if name_map:
            payload["tools"] = [self._to_anthropic_tool(k, v) for k, v in name_map.items()]

        client = self._ensure_client()
        async with client.stream(
            "POST",
            _API_URL,
            json=payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
        ) as resp:
            resp.raise_for_status()
            async for event in self._parse_sse(resp, name_map):
                yield event

    async def _parse_sse(
        self, resp: httpx.Response, name_map: dict[str, ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        # Tool-use blocks stream their input as `input_json_delta` fragments
        # keyed by content-block index; accumulate per index, then emit one
        # LLMToolCall when the block stops.
        tool_blocks: dict[int, dict] = {}
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            chunk = json.loads(data)
            # Anthropic streams mid-stream failures (overloaded_error,
            # api_error) as a 200-OK SSE `error` event, so raise_for_status
            # already passed. Surface it loudly — otherwise the turn ends with
            # empty text and a misleading `done` outcome, with nothing in the
            # trace naming why the cloud reply never came.
            if chunk.get("type") == "error":
                err = chunk.get("error") or {}
                raise RuntimeError(
                    f"anthropic stream error during chat ({self._model}): "
                    f"{err.get('type', 'unknown')}: {err.get('message', chunk)}"
                )
            for event in self._events_from_chunk(chunk, name_map, tool_blocks):
                yield event

    @staticmethod
    def _events_from_chunk(
        chunk: dict, name_map: dict[str, ToolSpec], tool_blocks: dict[int, dict]
    ) -> list[LLMEvent]:
        kind = chunk.get("type")
        if kind == "content_block_start":
            block = chunk.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_blocks[chunk["index"]] = {
                    "name": block.get("name", ""),
                    "json": "",
                }
            return []
        if kind == "content_block_delta":
            delta = chunk.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                return [LLMText(text=text)] if text else []
            if delta.get("type") == "input_json_delta":
                block = tool_blocks.get(chunk["index"])
                if block is not None:
                    block["json"] += delta.get("partial_json", "")
            return []
        if kind == "content_block_stop":
            block = tool_blocks.pop(chunk["index"], None)
            if block is None:
                return []
            return [AnthropicLLM._tool_call_from_block(block, name_map)]
        return []

    @staticmethod
    def _tool_call_from_block(
        block: dict, name_map: dict[str, ToolSpec]
    ) -> LLMToolCall:
        raw = block["json"].strip()
        try:
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        spec = name_map.get(block["name"])
        if spec is None:
            # Hallucinated name — surface it as a call with a sentinel server
            # so the registry returns "unknown tool" and the model can recover.
            return LLMToolCall(
                call_id=uuid.uuid4().hex,
                server="unknown",
                name=block["name"] or "unnamed",
                args=args,
            )
        return LLMToolCall(
            call_id=uuid.uuid4().hex, server=spec.server, name=spec.name, args=args
        )

    # ---- request translation ---------------------------------------------

    @staticmethod
    def _to_anthropic_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict]]:
        """Split our flat message list into Anthropic's (system, messages)
        shape. System messages fold into the top-level `system` string. Tool
        results become `tool_result` blocks inside a `user` message; runs of
        consecutive tool messages merge into one user turn so roles alternate
        as Anthropic requires."""
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
            elif m.role == "user":
                out.append({"role": "user", "content": m.content or ""})
            elif m.role == "assistant":
                out.append(
                    {"role": "assistant", "content": AnthropicLLM._assistant_blocks(m)}
                )
            elif m.role == "tool":
                AnthropicLLM._append_tool_result(out, m)
            else:
                raise ValueError(f"unknown role: {m.role}")
        return "\n\n".join(system_parts), out

    @staticmethod
    def _assistant_blocks(m: LLMMessage) -> list[dict]:
        blocks: list[dict] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.call_id,
                    "name": AnthropicLLM._sanitise_pair(tc.server, tc.name),
                    "input": tc.args,
                }
            )
        # An assistant turn must carry at least one block.
        return blocks or [{"type": "text", "text": ""}]

    @staticmethod
    def _append_tool_result(out: list[dict], m: LLMMessage) -> None:
        block = {
            "type": "tool_result",
            "tool_use_id": m.tool_call_id or "",
            "content": m.content or "",
        }
        # Merge into the immediately-preceding user turn if it's already a
        # tool-result batch, so multiple tool calls in one assistant step
        # return as a single user message.
        if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    @staticmethod
    def _to_anthropic_tool(sanitised_name: str, spec: ToolSpec) -> dict:
        return {
            "name": sanitised_name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }

    @staticmethod
    def _sanitise(spec: ToolSpec) -> str:
        return AnthropicLLM._sanitise_pair(spec.server, spec.name)

    @staticmethod
    def _sanitise_pair(server: str, name: str) -> str:
        if "__" in server or "__" in name:
            raise ValueError(
                f"server/tool names must not contain '__' (reserved separator): {server}.{name}"
            )
        return f"{server}__{name}"
