"""The per-utterance drive budget, enforced.

Commit `0613ebc` added a third re-drive (`_finish_the_job`) on top of the tool
scope fallback and the specialist escalation, and stated a worst case of four
drives -- roughly 32 LLM passes -- for a single utterance. Nothing checked it:
before this file, no test referenced `_finish_the_job` or `confabulation_retry`
at all, so the budget was a claim in a docstring.

It is worth a test because every one of those re-drives was added separately,
each bounded on its own, and the ceiling is the product of their interaction
rather than anything one of them states. A fourth path added later would raise
it silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from glados.core.adapters import LLMEvent, LLMMessage, LLMText, LLMToolCall, ToolSpec
from glados.mcp.registry import MCPCallResult, MCPRegistry

from .organizer_harness import CLIENT_ID, desk_organizer


class _ReadOnlyTool:
    """A successful, NON-mutating call. Enough to keep a turn out of
    `_confabulated` (which needs a zero-tool turn) while leaving
    `_action_drifted` satisfied -- the honest under-delivery shape."""

    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            server="cart", name=name, description="", parameters={}
        )

    async def call(self, args, envelope):
        return MCPCallResult(ok=True, content={"items": ["3L milk"]})


class _CountingLLM:
    """Counts drives and never lands a mutation.

    Replies with a TRUE statement about what it read, so no claim guard fires
    and the turn classifies `failed` via the drift check -- which is the
    predicate both the scope fallback and the escalation key on."""

    def __init__(self, *, confabulate: bool = False) -> None:
        self.drives = 0
        self._confabulate = confabulate

    async def chat(
        self, messages: list[LLMMessage], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        if messages[-1].role == "tool":
            yield LLMText(text="Cart has 1 x 3L milk.")
            return
        self.drives += 1
        if self._confabulate:
            # Zero tools plus a declarative claim -- the one shape that is
            # SAFE to replay, and so the only one `_finish_the_job` accepts.
            yield LLMText(text="Milk removed from cart.")
            return
        yield LLMToolCall(call_id="c1", server="cart", name="view", args={})


class _DroppingRouter:
    """Offers a strict subset, so `scoped` is True and the capability
    fallback is armed."""

    def scope_for(self, text: str, specs: list[ToolSpec]) -> list[ToolSpec]:
        return specs[:1]


def _read_only_registry() -> MCPRegistry:
    reg = MCPRegistry()
    reg.register(_ReadOnlyTool("view"))
    reg.register(_ReadOnlyTool("search"))
    return reg


@pytest.mark.asyncio
async def test_a_clean_turn_costs_exactly_one_drive(tmp_path: Path) -> None:
    """The budget only exists for failures. A turn that answers must not pay
    for the machinery that recovers the ones that don't."""
    llm = _CountingLLM()
    async with desk_organizer(
        tmp_path, llm=llm, mcp=_read_only_registry(), tool_router=_DroppingRouter()
    ) as h:
        await h.org.handle_user_text(CLIENT_ID, "what is in my cart?")
        await h.org.flush()
    assert llm.drives == 1


@pytest.mark.asyncio
async def test_worst_case_never_exceeds_four_drives(tmp_path: Path) -> None:
    """Scoped drift -> capability fallback -> specialist -> finish-the-job.

    Two drives land on each brain, which is easy to get wrong from reading the
    call order alone: the capability fallback re-drives the PRIMARY (it is a
    missing-tool retry, not a smarter-brain one), while `_finish_the_job` runs
    on `answered_by` -- the specialist, because escalation already swapped the
    brain that produced the outcome.

    The specialist confabulates so the last re-drive is actually reached; a
    specialist that merely drifted again would stop at three and quietly stop
    testing the thing this file is named after."""
    primary = _CountingLLM()
    specialist = _CountingLLM(confabulate=True)
    async with desk_organizer(
        tmp_path,
        llm=primary,
        mcp=_read_only_registry(),
        tool_router=_DroppingRouter(),
        specialist_llm=specialist,
    ) as h:
        await h.org.handle_user_text(CLIENT_ID, "remove the milk")
        await h.org.flush()
    total = primary.drives + specialist.drives
    assert total <= 4, f"drive budget blown: {total}"
    # Pinned per brain, not just as a sum: a different path costing the same
    # four would otherwise pass while no longer testing what this names.
    assert (primary.drives, specialist.drives) == (2, 2), (
        f"expected 2 primary drives and 2 specialist, got "
        f"{primary.drives} and {specialist.drives} -- if a re-drive was "
        "removed or re-gated, retire or update this test rather than "
        "loosening it"
    )


@pytest.mark.asyncio
async def test_a_landed_mutation_blocks_every_re_drive(tmp_path: Path) -> None:
    """The interlock the whole budget rests on: once external state has really
    changed, no path may replay the turn, or the side effect fires twice.

    The turn still classifies `failed` -- a later unrelated tool error is
    unrecovered -- so every re-drive is armed and must decline on the
    mutation alone. A FAILED write would not test this: nothing landed, so
    replay is safe and correctly allowed."""

    class _Write:
        spec = ToolSpec(
            server="cart", name="write", description="", parameters={},
            mutating=True,
        )

        async def call(self, args, envelope):
            return MCPCallResult(ok=True, content={"added": "milk"})

    class _Breaks:
        spec = ToolSpec(server="cart", name="search", description="", parameters={})

        async def call(self, args, envelope):
            return MCPCallResult(ok=False, error="upstream is down")

    class _WritesThenBreaks:
        """Lands the mutation, then trips over an unrelated tool."""

        def __init__(self) -> None:
            self.drives = 0
            self._calls = 0

        async def chat(self, messages, tools):
            if messages[-1].role == "tool":
                self._calls += 1
                if self._calls == 1:
                    yield LLMToolCall(
                        call_id="c2", server="cart", name="search", args={}
                    )
                    return
                yield LLMText(text="Cart has 1 x 3L milk.")
                return
            self.drives += 1
            yield LLMToolCall(call_id="c1", server="cart", name="write", args={})

    llm = _WritesThenBreaks()
    reg = MCPRegistry()
    reg.register(_Write())
    reg.register(_Breaks())
    async with desk_organizer(
        tmp_path,
        llm=llm,
        mcp=reg,
        tool_router=_DroppingRouter(),
        specialist_llm=_CountingLLM(),
    ) as h:
        await h.org.handle_user_text(CLIENT_ID, "remove the milk")
        await h.org.flush()
    assert llm.drives == 1
