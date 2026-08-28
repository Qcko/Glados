"""A timed-out mutating call is indeterminate, not failed.

Slice 2 of `DESIGN-dispatch-cancellation.md`. The transport half (slice 1)
stops the client lying to itself about an abandoned request; this half stops it
lying to the MODEL, and -- more importantly -- stops every replay path
concluding "nothing landed, safe to try again" about a cart write that may be
executing while it decides.

The prompt-level advice is here too, but it is not the control: these models
follow prompts unreliably, so the in-flight ledger refuses the duplicate in
code and the advisory only explains why.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.core.turn_outcome import (
    TurnRecord,
    claimed_a_change_it_did_not_make,
    classify,
)
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry


_ENVELOPE = CallEnvelope(session_id="s1", room_id="desk", speaker_id="qcko")


# ---- turn_outcome: the two questions are not the same question ----------


def test_an_indeterminate_mutation_may_have_landed() -> None:
    turn = TurnRecord(final_text="Something went wrong.")
    turn.record_tool("dunnes.add_to_cart", ok=False, mutating=True, indeterminate=True)
    assert not turn.made_successful_mutation()  # the goal was not achieved
    assert turn.may_have_mutated()  # but replaying it is not safe


def test_an_indeterminate_read_does_not_block_a_replay() -> None:
    turn = TurnRecord(final_text="Something went wrong.")
    turn.record_tool("dunnes.view_cart", ok=False, mutating=False, indeterminate=True)
    assert not turn.may_have_mutated()


def test_an_indeterminate_turn_still_classifies_failed() -> None:
    """Truthful classification plus a hard replay gate is the conservative
    pairing. Reporting anything else would claim success for a write that may
    never have happened."""
    turn = TurnRecord(final_text="I could not tell whether that worked.")
    turn.record_tool("dunnes.add_to_cart", ok=False, mutating=True, indeterminate=True)
    assert classify(turn) == "failed"


def test_claim_check_fails_open_on_an_indeterminate_mutation() -> None:
    """The reply says the milk was removed and the dispatch record shows no
    successful mutating call -- but the one that timed out may have removed it.
    Accusing here tells the user their shopping did not happen when it did."""
    turn = TurnRecord(final_text="Milk removed from cart.")
    turn.record_tool(
        "dunnes.remove_from_cart",
        ok=False,
        mutating=True,
        indeterminate=True,
        args={"item": "milk"},
    )
    assert not claimed_a_change_it_did_not_make(turn)


def test_an_indeterminate_call_excuses_only_its_own_subject() -> None:
    """It must not become a blanket amnesty for the turn: one outstanding call
    excusing every invented claim beside it is the failure the per-clause
    check was built to stop."""
    turn = TurnRecord(final_text="Added the milk and removed the eggs.")
    turn.record_tool(
        "dunnes.add_to_cart",
        ok=False,
        mutating=True,
        indeterminate=True,
        args={"item": "milk"},
    )
    turn.record_tool(
        "dunnes.remove_from_cart", ok=False, mutating=True, args={"item": "eggs"}
    )
    assert claimed_a_change_it_did_not_make(turn)


def test_claim_check_still_fires_on_a_plain_failure() -> None:
    turn = TurnRecord(final_text="Milk removed from cart.")
    turn.record_tool("dunnes.remove_from_cart", ok=False, mutating=True)
    assert claimed_a_change_it_did_not_make(turn)


# ---- registry: the dispatcher states the transport fact ------------------


class _HangingTool:
    """Never answers. `spec.timeout_s` is what ends the call."""

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec
        self.calls: list[dict] = []

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(args)
        await asyncio.sleep(30)
        return MCPCallResult(ok=True)


class _RaisingTool:
    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        raise RuntimeError("boom")


def _spec(**kwargs) -> ToolSpec:
    base = dict(
        server="dunnes",
        name="add_to_cart",
        description="add",
        parameters={"type": "object"},
    )
    base.update(kwargs)
    return ToolSpec(**base)


async def test_dispatch_marks_a_timeout_indeterminate() -> None:
    mcp = MCPRegistry()
    mcp.register(_HangingTool(_spec(timeout_s=0.05, mutating=True)))
    result = await mcp.dispatch(
        "dunnes", "add_to_cart", {"item": "milk"}, _ENVELOPE
    )
    assert not result.ok
    assert result.indeterminate


async def test_a_tool_that_raised_is_a_plain_failure() -> None:
    """Nothing was left running, so this one is safe to retry -- the
    distinction the whole slice turns on."""
    mcp = MCPRegistry()
    mcp.register(_RaisingTool(_spec(mutating=True)))
    result = await mcp.dispatch("dunnes", "add_to_cart", {}, _ENVELOPE)
    assert not result.ok
    assert not result.indeterminate


# ---- organizer: the ledger, and where the advisory is delivered ----------


class _ScriptedLLM:
    """Emits a scripted list of tool calls, one pass each, then a reply."""

    def __init__(self, calls: list[LLMToolCall], reply: str = "All done.") -> None:
        self._calls = calls
        self._reply = reply
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        index = len(self.passes) - 1
        if index < len(self._calls):
            yield self._calls[index]
            return
        yield LLMText(text=self._reply)


def _call(call_id: str, args: dict) -> LLMToolCall:
    return LLMToolCall(
        call_id=call_id, server="dunnes", name="add_to_cart", args=args
    )


@asynccontextmanager
async def _organizer(tmp: Path, llm, mcp: MCPRegistry, **kwargs):
    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    org = Organizer(
        llm=llm,
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=lambda cid: binding if cid == "desk-ui" else None,
        clients_in_room=lambda rid: ["desk-ui"] if rid == "desk" else [],
        **kwargs,
    )
    try:
        yield org, sink
    finally:
        await org.close()


def _tool_messages(passes: list[list[LLMMessage]]) -> list[str]:
    last = passes[-1]
    return [m.content or "" for m in last if m.role == "tool"]


async def test_a_reissued_call_never_reaches_the_wire(tmp_path: Path) -> None:
    """The model is told the write failed, so it retries -- which is the
    duplicate cart line. The second dispatch is answered from the ledger."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True, untrusted=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM(
        [_call("c1", {"item": "milk"}), _call("c2", {"item": "milk"})]
    )
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    assert len(tool.calls) == 1
    assert "already outstanding" in _tool_messages(llm.passes)[-1]


async def test_a_ledger_answer_is_not_dressed_as_external_content(
    tmp_path: Path,
) -> None:
    """GLaDOS wrote the refusal and it never went near the wire. Wrapping it
    would put our own words in the region the model is told to ignore -- and
    would set the session-sticky `untrusted_seen` on a call that ingested no
    bytes at all."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True, untrusted=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM([_call("c1", {"item": "milk"}), _call("c2", {"item": "milk"})])
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    refusal = _tool_messages(llm.passes)[-1]
    assert "already outstanding" in refusal
    assert "<external>" not in refusal


async def test_a_different_call_is_not_blocked(tmp_path: Path) -> None:
    """The ledger is keyed on the call, not on the tool: a genuinely different
    request is still allowed through while one is outstanding."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM([_call("c1", {"item": "milk"}), _call("c2", {"item": "eggs"})])
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    assert [c["item"] for c in tool.calls] == ["milk", "eggs"]


async def test_the_advisory_lands_outside_the_external_wrapper(
    tmp_path: Path,
) -> None:
    """The system prompt tells the model that anything inside `<external>` is
    data rather than instructions, so an instruction delivered in there is one
    it has been told to ignore."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True, untrusted=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM([_call("c1", {"item": "milk"})])
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    message = _tool_messages(llm.passes)[-1]
    assert "GLaDOS note" in message
    assert message.index("</external>") < message.index("GLaDOS note")


async def test_a_read_that_timed_out_gets_no_advisory(tmp_path: Path) -> None:
    """Re-issuing a read is harmless, and a note on every slow search is noise
    that teaches the model to skim past it."""
    tool = _HangingTool(_spec(name="search", timeout_s=0.05, untrusted=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM(
        [LLMToolCall(call_id="c1", server="dunnes", name="search", args={"q": "milk"})]
    )
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "find milk")
        await org.flush()

    assert "GLaDOS note" not in _tool_messages(llm.passes)[-1]


async def test_an_indeterminate_mutation_blocks_specialist_escalation(
    tmp_path: Path,
) -> None:
    """The turn classifies `failed`, which is the escalation trigger -- and
    re-driving it cold on the specialist is exactly how one "add milk" becomes
    two."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    primary = _ScriptedLLM([_call("c1", {"item": "milk"})], reply="I think so?")
    specialist = _ScriptedLLM([], reply="Added milk.")
    async with _organizer(
        tmp_path, primary, mcp, specialist_llm=specialist, escalate_on_failed=True
    ) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    assert specialist.passes == []
    assert len(tool.calls) == 1


async def test_a_plain_failure_still_escalates(tmp_path: Path) -> None:
    """The gate must not have become "never escalate a failed turn"."""
    mcp = MCPRegistry()
    mcp.register(_RaisingTool(_spec(mutating=True)))
    primary = _ScriptedLLM([_call("c1", {"item": "milk"})], reply="I think so?")
    specialist = _ScriptedLLM([], reply="Added milk.")
    async with _organizer(
        tmp_path, primary, mcp, specialist_llm=specialist, escalate_on_failed=True
    ) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    assert specialist.passes != []


@pytest.mark.parametrize("args", [{"a": 1, "b": 2}, {"b": 2, "a": 1}])
async def test_the_ledger_key_ignores_argument_order(
    tmp_path: Path, args: dict
) -> None:
    """Key order is not a different request, and a model re-emitting a call is
    under no obligation to serialise it the same way twice."""
    tool = _HangingTool(_spec(timeout_s=0.05, mutating=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _ScriptedLLM([_call("c1", {"a": 1, "b": 2}), _call("c2", args)])
    async with _organizer(tmp_path, llm, mcp) as (org, _):
        await org.handle_user_text("desk-ui", "add milk")
        await org.flush()

    assert len(tool.calls) == 1
