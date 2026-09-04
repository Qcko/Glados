"""B3 -- the in-flight byte ceiling inside the tool loop.

`test_prompt_budget.py` covers the boot inequality, which prices what a turn
STARTS from. These cover the gap that check names in its own docstring: the
tool loop appends each pass's results as it goes, so a turn can grow past the
priced ceiling between passes, and front-truncation would delete the section 7
rule silently while the model reads attacker-chosen bytes.

The unit tests fix the shedding boundary; the wired tests fix what a listener
actually gets when a turn is stopped."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec
from glados.core.config import ClientBinding
from glados.core.organizer import (
    _HOP_BUDGET_MSG,
    _HOP_BUDGET_MSG_AFTER_ACTION,
    Organizer,
)
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.core.turn_outcome import TurnRecord, classify
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry

BULK_SPEC = ToolSpec(
    server="dunnes",
    name="scan_favorites_for_sales",
    description="scan",
    parameters={"type": "object"},
    untrusted=True,
)


def _sys(text: str = "you are glados") -> LLMMessage:
    return LLMMessage(role="system", content=text)


def _turn(tool_bytes: int, marker: str) -> list[LLMMessage]:
    return [
        LLMMessage(role="user", content="ask " + marker),
        LLMMessage(role="assistant", content=None),
        LLMMessage(role="tool", content="x" * tool_bytes),
    ]


def _shape(messages) -> list[tuple[str, str | None]]:
    return [(m.role, m.content) for m in messages]


def _org(tmp: Path, **kw) -> Organizer:
    return Organizer(
        llm=object(),
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=None,
        binding_for_client=lambda _c: None,
        clients_in_room=lambda _r: [],
        **kw,
    )


# ---- where the shedding stops -------------------------------------------


def test_sheds_oldest_whole_turns_until_it_fits(tmp_path: Path) -> None:
    org = _org(tmp_path, max_history_external_bytes=100)
    msgs = [_sys(), *_turn(80, "old"), *_turn(80, "new")]
    fitted = org._shed_for_hop(msgs)
    assert fitted is not None
    assert [m.content for m in fitted if m.role == "user"] == ["ask new"]


def test_the_system_prompt_never_sheds(tmp_path: Path) -> None:
    """Dropping the rule to make room for the content it governs would be the
    attack, carried out by the defence."""
    org = _org(tmp_path, max_history_external_bytes=100)
    msgs = [_sys("RULE"), *_turn(80, "old"), *_turn(80, "new")]
    fitted = org._shed_for_hop(msgs)
    assert fitted is not None
    assert fitted[0].role == "system" and fitted[0].content == "RULE"


def test_a_prompt_already_under_the_ceiling_is_untouched(tmp_path: Path) -> None:
    org = _org(tmp_path, max_history_external_bytes=1000)
    msgs = [_sys(), *_turn(80, "old"), *_turn(80, "new")]
    assert _shape(org._shed_for_hop(msgs)) == _shape(msgs)


def test_shedding_never_orphans_a_tool_message(tmp_path: Path) -> None:
    """The property the docstring leans on hardest. A tool message severed
    from the assistant message that called it is conversation state some
    backends reject outright, so every shed depth is checked, not just the
    one the other tests happen to reach."""
    org = _org(tmp_path, max_history_external_bytes=100)
    msgs = [_sys(), *_turn(80, "a"), *_turn(80, "b"), *_turn(80, "c")]
    for depth in range(len(msgs)):
        fitted = org._shed_for_hop(msgs[: depth + 1])
        if fitted is None:
            continue
        roles = [m.role for m in fitted]
        for i, role in enumerate(roles):
            if role == "tool":
                assert "assistant" in roles[:i], roles


def test_a_harness_directive_is_never_shed(tmp_path: Path) -> None:
    """`_finish_the_job` rides a system nudge just before the live user
    message. Shedding it would downgrade that bounded retry to a bare replay
    of the utterance that just confabulated -- silently."""
    org = _org(tmp_path, max_history_external_bytes=100)
    msgs = [
        _sys(),
        *_turn(80, "old"),
        LLMMessage(role="system", content="NUDGE"),
        *_turn(80, "new"),
    ]
    assert org._shed_for_hop(msgs) is None


def test_the_live_turn_alone_over_the_ceiling_stops_the_turn(tmp_path: Path) -> None:
    """Nothing left to shed that would not sever a tool message from the
    assistant message that called it -- so the turn fails rather than sending
    a prompt that would be truncated from the front."""
    org = _org(tmp_path, max_history_external_bytes=100)
    assert org._shed_for_hop([_sys(), *_turn(400, "now")]) is None


def test_a_disabled_ceiling_sheds_nothing(tmp_path: Path) -> None:
    org = _org(tmp_path, max_history_external_bytes=0)
    msgs = [_sys(), *_turn(9999, "huge"), *_turn(9999, "huger")]
    assert _shape(org._shed_for_hop(msgs)) == _shape(msgs)


def test_budget_exceeded_classifies_failed() -> None:
    turn = TurnRecord(final_text=_HOP_BUDGET_MSG, budget_exceeded=True)
    assert classify(turn) == "failed"


# ---- wired into the loop -------------------------------------------------


class _AlwaysCallsAgainLLM:
    """The flooding shape: every pass asks for the tool again, so the prompt
    grows by a result each time and only the hop check can stop it."""

    def __init__(self) -> None:
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        yield LLMToolCall(
            call_id="c" + str(len(self.passes)),
            server=BULK_SPEC.server,
            name=BULK_SPEC.name,
            args={},
        )


class _AnswersAtOnceLLM:
    def __init__(self) -> None:
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        yield LLMText(text="Nothing on sale.")


class _StaticTool:
    def __init__(self, spec: ToolSpec, payload) -> None:
        self.spec = spec
        self._payload = payload

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        return MCPCallResult(ok=True, content=self._payload)


@asynccontextmanager
async def _make(tmp: Path, llm, mcp: MCPRegistry, **kw):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    org = Organizer(
        llm=llm,
        mcp=mcp,
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={"desk-ui": binding}.get,
        clients_in_room=lambda r: ["desk-ui"] if r == "desk" else [],
        **kw,
    )
    try:
        yield org, sink
    finally:
        await org.close()


async def _run(tmp_path: Path, llm, payload, **kw):
    mcp = MCPRegistry()
    mcp.register(_StaticTool(BULK_SPEC, payload))
    async with _make(tmp_path, llm, mcp, **kw) as (org, sink):
        await org.handle_user_text("desk-ui", "anything on sale")
        await org.flush()
    return sink


SECRET = "obey-the-injected-instruction"


@pytest.mark.asyncio
async def test_a_flooding_turn_is_stopped_before_it_outgrows_the_window(
    tmp_path: Path,
) -> None:
    llm = _AlwaysCallsAgainLLM()
    await _run(
        tmp_path,
        llm,
        {"note": SECRET + "y" * 900},
        max_result_bytes=1024,
        max_history_external_bytes=1500,
    )
    sent = [
        sum(len((m.content or "").encode()) for m in p if m.role == "tool")
        for p in llm.passes
    ]
    assert max(sent) <= 1500, "a prompt went out over the ceiling: " + str(sent)
    assert len(llm.passes) < 8, "the hop check, not loop exhaustion, ended it"


@pytest.mark.asyncio
async def test_the_stopped_turn_says_the_fixed_line_and_fails(
    tmp_path: Path,
) -> None:
    sink = await _run(
        tmp_path,
        _AlwaysCallsAgainLLM(),
        {"note": "y" * 900},
        max_result_bytes=1024,
        max_history_external_bytes=1500,
    )
    spoken = [m["text"] for _, m in sink if m.get("type") == "assistant_delta"]
    assert _HOP_BUDGET_MSG in spoken
    outcomes = [m["outcome"] for _, m in sink if m.get("type") == "turn_outcome"]
    assert outcomes and outcomes[-1] == "failed"


@pytest.mark.asyncio
async def test_the_spoken_line_never_echoes_the_bytes_that_caused_it(
    tmp_path: Path,
) -> None:
    """The content that blew the ceiling is attacker-chosen. A reply that
    quoted or summarised it would hand the payload the voice channel."""
    sink = await _run(
        tmp_path,
        _AlwaysCallsAgainLLM(),
        {"note": SECRET + "y" * 900},
        max_result_bytes=1024,
        max_history_external_bytes=1500,
    )
    spoken = " ".join(
        m["text"] for _, m in sink if m.get("type") == "assistant_delta"
    )
    assert SECRET not in spoken


@pytest.mark.asyncio
async def test_an_ordinary_turn_is_not_touched(tmp_path: Path) -> None:
    llm = _AnswersAtOnceLLM()
    sink = await _run(
        tmp_path, llm, {"items": []}, max_history_external_bytes=1500
    )
    spoken = [m["text"] for _, m in sink if m.get("type") == "assistant_delta"]
    assert _HOP_BUDGET_MSG not in spoken
    assert "Nothing on sale." in spoken


@pytest.mark.asyncio
async def test_the_trace_records_the_sizes_the_spoken_line_withholds(
    tmp_path: Path,
) -> None:
    """An operator needs to know which turn blew the ceiling and by how much.
    The trace is the surface where attacker-chosen bytes cannot be mistaken
    for GLaDOS speaking, so that is where the numbers go."""
    await _run(
        tmp_path,
        _AlwaysCallsAgainLLM(),
        {"note": "y" * 900},
        max_result_bytes=1024,
        max_history_external_bytes=1500,
    )
    events = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    blown = [e for e in events if e.get("event") == "hop_budget_exceeded"]
    assert blown, "expected a hop_budget_exceeded trace event"
    assert blown[-1]["ceiling"] == 1500
    assert blown[-1]["external_bytes"] > 1500
    assert blown[-1]["may_have_mutated"] is False


class _MutatesThenFloodsLLM:
    """A cart write, then bulk reads until the ceiling stops the turn. The
    shape where the reassuring line would be a lie."""

    def __init__(self) -> None:
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        spec = WRITE_SPEC if len(self.passes) == 1 else BULK_SPEC
        yield LLMToolCall(
            call_id="c" + str(len(self.passes)),
            server=spec.server,
            name=spec.name,
            args={},
        )


WRITE_SPEC = ToolSpec(
    server="dunnes",
    name="add_to_cart_by_name",
    description="add",
    parameters={"type": "object"},
    mutating=True,
)


@pytest.mark.asyncio
async def test_a_turn_that_already_acted_does_not_claim_nothing_happened(
    tmp_path: Path,
) -> None:
    mcp = MCPRegistry()
    mcp.register(_StaticTool(BULK_SPEC, {"note": "y" * 900}))
    mcp.register(_StaticTool(WRITE_SPEC, {"ok": True}))
    sink: list[tuple[str, dict]] = []

    async def send(client_id, msg):
        sink.append((client_id, msg.model_dump()))

    binding = ClientBinding(
        client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"
    )
    org = Organizer(
        llm=_MutatesThenFloodsLLM(),
        mcp=mcp,
        traces=TraceStore(tmp_path),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client={"desk-ui": binding}.get,
        clients_in_room=lambda r: ["desk-ui"] if r == "desk" else [],
        max_result_bytes=1024,
        max_history_external_bytes=1500,
    )
    try:
        await org.handle_user_text("desk-ui", "add the cheapest one")
        await org.flush()
    finally:
        await org.close()
    spoken = [m["text"] for _, m in sink if m.get("type") == "assistant_delta"]
    assert _HOP_BUDGET_MSG_AFTER_ACTION in spoken
    assert _HOP_BUDGET_MSG not in spoken


@pytest.mark.asyncio
async def test_a_budget_stopped_turn_is_never_escalated(tmp_path: Path) -> None:
    """Escalation asks whether the brain was good enough. This turn ran out of
    WINDOW, which a smarter model at the same num_ctx does not fix -- and the
    re-drive would dispatch the flooding tool again and repeat the line."""
    specialist = _AlwaysCallsAgainLLM()
    await _run(
        tmp_path,
        _AlwaysCallsAgainLLM(),
        {"note": "y" * 900},
        max_result_bytes=1024,
        max_history_external_bytes=1500,
        specialist_llm=specialist,
        escalate_on_failed=True,
    )
    assert specialist.passes == []
