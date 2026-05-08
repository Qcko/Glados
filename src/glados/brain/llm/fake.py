"""Deterministic fake LLM. Two behaviours:

- Last message is `user` and mentions "time" with `time.now` available →
  emit a tool_call.
- Last message is `tool` (a tool result) → emit a final text answer.
- Else → echo the user text.

Real Ollama / vLLM adapter swaps in later behind the same `LLM` Protocol.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from ...core.adapters import (
    LLMEvent,
    LLMMessage,
    LLMText,
    LLMToolCall,
    ToolSpec,
)


class FakeLLM:
    async def chat(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        last = messages[-1]

        if last.role == "tool":
            yield LLMText(text=f"It's {self._humanise(last.content)}.")
            return

        if last.role == "user":
            text = (last.content or "").lower()
            if "time" in text and any(t.qualified == "time.now" for t in tools):
                yield LLMToolCall(
                    call_id=uuid.uuid4().hex[:8],
                    server="time",
                    name="now",
                    args={},
                )
                return
            yield LLMText(text=f"echo: {last.content}")

    @staticmethod
    def _humanise(content: str | None) -> str:
        if not content:
            return ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content
        return data.get("human") or data.get("iso") or content
