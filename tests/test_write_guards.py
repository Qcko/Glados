"""Two harness guards on cart writes (DESIGN-write-guards.md).

Observed 11-09-2026: "add tomatoes to the cart" said twice became remove +
add four. The server's identical-write refusal saw two different calls, and
the per-turn in-flight ledger saw two different turns. The guards here refuse
in code what no prompt reliably prevents on a small local model: a count the
user never said (quantity provenance), and a re-issue of an add that already
landed (the cross-turn write ledger).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec
from glados.core.turn_outcome import (
    TurnRecord,
    claimed_a_change_it_did_not_make,
    classify,
)
from glados.core.config import ServerEntry, ToolOverlay
from glados.core.utterance import has_quantity_cue, has_repeat_cue, is_add_request
from glados.core.write_ledger import WriteLedger, canonical_key, coerce_quantity
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry
from tests.organizer_harness import CLIENT_ID, desk_organizer, trace_events

# ---- utterance cues ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("add tomatoes to the cart", False),
        ("add two tomatoes", True),
        ("add 2 tomatoes", True),
        ("add a few tomatoes", True),
        ("add some tomatoes", True),
        ("add more tomatoes", True),
        ("set tomatoes to zero", True),
        # Word-bounded: the count words hide inside ordinary nouns.
        ("add onions", False),
        ("add the phone charger", False),
        ("add tender stem broccoli", False),
        # A digit glued to a unit names a product, not a count.
        ("add 7up", False),
        ("add 1L milk", False),
        ("", False),
    ],
)
def test_quantity_cue(text: str, expected: bool) -> None:
    assert has_quantity_cue(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("add another tomatoes", True),
        ("more tomatoes please", True),
        ("do that again", True),
        ("add tomatoes to the cart", False),
        ("add a moreish snack", False),
    ],
)
def test_repeat_cue(text: str, expected: bool) -> None:
    assert has_repeat_cue(text) is expected


# ---- the ledger itself ------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _call(name: str, args: dict, call_id: str = "c1") -> LLMToolCall:
    return LLMToolCall(call_id=call_id, server="dunnes", name=name, args=args)


def test_key_ignores_case_whitespace_order_and_dropped_args() -> None:
    a = canonical_key(_call("add", {"query": "Tomatoes ", "quantity": 4, "repeat": True}), ["quantity", "repeat"])
    b = canonical_key(_call("add", {"quantity": 1, "query": "tomatoes"}), ["quantity", "repeat"])
    assert a == b


def test_key_keeps_the_subject_distinct() -> None:
    a = canonical_key(_call("add", {"query": "tomatoes"}))
    b = canonical_key(_call("add", {"query": "eggs"}))
    assert a != b


@pytest.mark.parametrize("raw, expected", [(4, 4), ("4", 4), (4.0, 4), (None, None), (True, None), ("x", None)])
def test_coerce_quantity(raw: object, expected: int | None) -> None:
    assert coerce_quantity(raw) == expected


def test_ledger_remembers_inside_the_window_and_forgets_after() -> None:
    clock = _Clock()
    ledger = WriteLedger(window_s=120, clock=clock)
    call = _call("add", {"query": "tomatoes"})
    key = canonical_key(call)
    ledger.note("s1", call, key, 1, certain=True)
    clock.now += 119
    assert ledger.recent("s1", key) is not None
    clock.now += 2
    assert ledger.recent("s1", key) is None


def test_ledger_is_per_session() -> None:
    ledger = WriteLedger(clock=_Clock())
    call = _call("add", {"query": "tomatoes"})
    ledger.note("s1", call, canonical_key(call), 1, certain=True)
    assert ledger.recent("s2", canonical_key(call)) is None


def test_ledger_clear_and_forget_drop_the_session() -> None:
    ledger = WriteLedger(clock=_Clock())
    call = _call("add", {"query": "tomatoes"})
    key = canonical_key(call)
    ledger.note("s1", call, key, 1, certain=True)
    ledger.clear("s1", "dunnes")
    assert ledger.recent("s1", key) is None
    ledger.note("s1", call, key, 1, certain=True)
    ledger.forget("s1")
    assert ledger.recent("s1", key) is None


def test_ledger_clear_is_per_server() -> None:
    """An intercom message or a timer says nothing about the cart."""
    ledger = WriteLedger(clock=_Clock())
    call = _call("add", {"query": "tomatoes"})
    key = canonical_key(call)
    ledger.note("s1", call, key, 1, certain=True)
    ledger.clear("s1", "room")
    assert ledger.recent("s1", key) is not None


def test_ledger_evicts_the_idlest_session() -> None:
    ledger = WriteLedger(clock=_Clock(), max_sessions=2)
    call = _call("add", {"query": "tomatoes"})
    key = canonical_key(call)
    ledger.note("s1", call, key, 1, certain=True)
    ledger.note("s2", call, key, 1, certain=True)
    ledger.note("s3", call, key, 1, certain=True)
    assert ledger.recent("s1", key) is None
    assert ledger.recent("s3", key) is not None


# ---- turn_outcome: a ledger answer meets the goal without lying ------------


def _satisfied_turn(final_text: str) -> TurnRecord:
    turn = TurnRecord(final_text=final_text, action_intent=True)
    turn.record_tool(
        "dunnes.add_to_cart_by_name",
        ok=True,
        mutating=False,
        args={"query": "tomatoes"},
        satisfied=True,
    )
    return turn


def test_a_satisfied_turn_is_done_whether_it_asks_or_states() -> None:
    assert classify(_satisfied_turn("Tomatoes are already in the cart. Want more?")) == "done"
    assert classify(_satisfied_turn("The tomatoes are already in your cart.")) == "done"


def test_a_satisfied_turn_did_not_mutate() -> None:
    """Every replay gate keeps asking the truthful question."""
    turn = _satisfied_turn("Already there.")
    assert not turn.may_have_mutated()


def test_a_satisfied_call_excuses_only_its_own_subject() -> None:
    turn = _satisfied_turn("Tomatoes are already in the cart. I added milk too.")
    assert claimed_a_change_it_did_not_make(turn)
    honest = _satisfied_turn("Tomatoes are in the cart. Added the tomatoes.")
    assert not claimed_a_change_it_did_not_make(honest)


def test_a_quantity_refusal_leaves_the_goal_unmet() -> None:
    """The guard refused a guess; asking is the right reply and classifies as
    a hand-back, not success."""
    turn = TurnRecord(final_text="How many tomatoes would you like?", action_intent=True)
    turn.record_tool("dunnes.add_to_cart_by_name", ok=True, mutating=False, args={"query": "tomatoes"})
    assert classify(turn) == "needs-user"


# ---- organizer: the guards in the dispatch path ----------------------------


class _RecordingTool:
    def __init__(self, spec: ToolSpec, result: MCPCallResult | None = None) -> None:
        self.spec = spec
        self.calls: list[dict] = []
        self._result = result or MCPCallResult(ok=True, content={"added": 1})

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(dict(args))
        return self._result


def _add_spec(**kwargs) -> ToolSpec:
    base = dict(
        server="dunnes",
        name="add_to_cart_by_name",
        description="add",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "quantity": {"type": "integer"}, "repeat": {"type": "boolean"}},
        },
        mutating=True,
        additive=True,
        quantity_arg="quantity",
    )
    base.update(kwargs)
    return ToolSpec(**base)


def _remove_spec() -> ToolSpec:
    return ToolSpec(
        server="dunnes",
        name="remove_from_cart",
        description="remove",
        parameters={"type": "object", "properties": {"productId": {"type": "string"}}},
        mutating=True,
        removes=True,
    )


class _TurnScriptedLLM:
    """One script per turn: a list of tool calls, then a reply."""

    def __init__(self, turns: list[list[LLMToolCall]], reply: str = "Done.") -> None:
        self._turns = turns
        self._reply = reply
        self.passes: list[list[LLMMessage]] = []
        self._turn = -1
        self._pass = 0

    def next_turn(self) -> None:
        self._turn += 1
        self._pass = 0

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        calls = self._turns[self._turn]
        if self._pass < len(calls):
            call = calls[self._pass]
            self._pass += 1
            yield call.model_copy(deep=True)
            return
        yield LLMText(text=self._reply)


def _add(query: str, call_id: str, **extra) -> LLMToolCall:
    return _call("add_to_cart_by_name", {"query": query, **extra}, call_id)


def _last_tool_message(llm: _TurnScriptedLLM) -> str:
    return [m.content or "" for m in llm.passes[-1] if m.role == "tool"][-1]


async def _say(h, llm: _TurnScriptedLLM, text: str) -> None:
    llm.next_turn()
    await h.org.handle_user_text(CLIENT_ID, text)
    await h.org.flush()


async def test_an_invented_quantity_is_refused_before_the_ledger_answers(tmp_path: Path) -> None:
    """Invariant 1: guard 1 wins, so the refusal asks for a count instead of
    laundering the four into "already done"."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [_add("tomatoes", "c2", quantity=4)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 1
    refusal = _last_tool_message(llm)
    assert "quantity_needed" in refusal
    assert "<external>" not in refusal


async def test_an_exact_repeat_is_already_done(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [_add("Tomatoes", "c2", quantity=1)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 1
    assert "already_done" in _last_tool_message(llm)
    events = [e.get("event") for e in trace_events(tmp_path)]
    assert "repeat_refused" in events


async def test_a_repeat_cue_lets_the_add_through_with_repeat_set(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [_add("tomatoes", "c2")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add another tomatoes")

    assert [c["repeat"] for c in tool.calls] == [False, True]


async def test_the_model_cannot_set_repeat_itself(tmp_path: Path) -> None:
    """`repeat: true` from the model is overwritten from the user's words, and
    it does not mint a new ledger key either."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1", repeat=True)], [_add("tomatoes", "c2", repeat=True)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert [c["repeat"] for c in tool.calls] == [False]
    assert "already_done" in _last_tool_message(llm)


async def test_a_counted_add_is_allowed(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1", quantity=4)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add four tomatoes")

    assert tool.calls[0]["quantity"] == 4


async def test_a_remove_clears_the_ledger(tmp_path: Path) -> None:
    add = _RecordingTool(_add_spec())
    remove = _RecordingTool(_remove_spec(), MCPCallResult(ok=True, content={"removed": 1}))
    mcp = MCPRegistry()
    mcp.register(add)
    mcp.register(remove)
    llm = _TurnScriptedLLM(
        [
            [_add("tomatoes", "c1")],
            [_call("remove_from_cart", {"productId": "1"}, "c2")],
            [_add("tomatoes", "c3")],
        ]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "remove the tomatoes")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(add.calls) == 2


async def test_a_timed_out_add_is_outcome_unknown_next_turn(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec(), MCPCallResult(ok=False, indeterminate=True, error="timeout"))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [_add("tomatoes", "c2")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 1
    assert "outcome_unknown" in _last_tool_message(llm)


async def test_the_window_expires(tmp_path: Path) -> None:
    clock = _Clock()
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [_add("tomatoes", "c2")]])
    ledger = WriteLedger(window_s=120, clock=clock)
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False, write_ledger=ledger) as h:
        await _say(h, llm, "add tomatoes to the cart")
        clock.now += 121
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 2


async def test_guards_stand_down_off_english(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("rajcata", "c1")], [_add("rajcata", "c2", quantity=4)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False, reply_language="cs") as h:
        await _say(h, llm, "pridej rajcata")
        await _say(h, llm, "pridej rajcata")

    assert len(tool.calls) == 2
    assert "repeat" not in tool.calls[0]


async def test_a_ledger_answer_does_not_gate_the_session(tmp_path: Path) -> None:
    """The refusal never went to the wire, so it must not mark the session as
    having read untrusted bytes. The ledger is seeded directly so the only
    turn is the refused one."""
    tool = _RecordingTool(_add_spec(untrusted=True))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[], [_add("tomatoes", "c1")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "hello")
        sid = h.session_id()
        seed = _add("tomatoes", "seed")
        h.org._write_ledger.note(sid, seed, canonical_key(seed, ["quantity", "repeat"]), 1, certain=True)
        await _say(h, llm, "add tomatoes to the cart")
        gated = sid in h.org._untrusted_sessions

    assert tool.calls == []
    assert "already_done" in _last_tool_message(llm)
    assert not gated


async def test_another_servers_write_does_not_clear_the_ledger(tmp_path: Path) -> None:
    """An intercom message between two adds says nothing about the cart."""
    add = _RecordingTool(_add_spec())
    other = _RecordingTool(
        ToolSpec(server="timer", name="set", description="t", parameters={"type": "object"}, mutating=True)
    )
    mcp = MCPRegistry()
    mcp.register(add)
    mcp.register(other)
    llm = _TurnScriptedLLM(
        [
            [_add("tomatoes", "c1")],
            [LLMToolCall(call_id="c2", server="timer", name="set", args={"minutes": 5})],
            [_add("tomatoes", "c3")],
        ]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "set a timer for five minutes")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(add.calls) == 1


async def test_a_string_quantity_is_still_a_guess(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1", quantity="4")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")

    assert tool.calls == []
    assert "quantity_needed" in _last_tool_message(llm)


async def test_a_refused_repeat_never_reaches_the_specialist(tmp_path: Path) -> None:
    """Invariant 4: the refusal is ok=True, so the turn is not `failed` and the
    escalation path that would re-issue the same call is never taken."""

    class _NeverCalled:
        calls = 0

        async def chat(self, messages, tools):
            _NeverCalled.calls += 1
            yield LLMText(text="specialist")

    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM(
        [[_add("tomatoes", "c1")], [_add("tomatoes", "c2")]],
        reply="The tomatoes are already in your cart.",
    )
    async with desk_organizer(
        tmp_path, llm=llm, mcp=mcp, escalate_on_failed=True, specialist_llm=_NeverCalled()
    ) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 1
    assert _NeverCalled.calls == 0


async def test_start_over_forgets_the_ledger(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("tomatoes", "c1")], [], [_add("tomatoes", "c3")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "start over")
        await _say(h, llm, "add tomatoes to the cart")

    assert len(tool.calls) == 2


# ---- guard 3: an add request must not remove --------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("add tomatoes to the cart", True),
        ("please buy milk", True),
        ("actually, put bread in the cart", True),
        ("order some eggs", True),
        ("remove the tomatoes", False),
        ("add eggs instead of the milk", False),
        ("add eggs and take the milk off", False),
        ("replace the milk with oat milk", False),
        ("add eggs and cancel the milk", False),
        ("add eggs and set milk to 0", False),
        ("add eggs, no milk", False),
        ("add tomatoes not onions", False),
        ("add eggs and scrap the milk", False),
        ("add some offers", True),
        ("tell me what to add", False),
        ("what's in my cart", False),
        ("", False),
    ],
)
def test_add_request(text: str, expected: bool) -> None:
    assert is_add_request(text) is expected


def test_a_removal_arg_without_removes_is_a_config_error() -> None:
    with pytest.raises(ValueError):
        ToolOverlay(mutating=True, delta_arg="delta")


def test_count_and_delta_together_is_a_config_error() -> None:
    with pytest.raises(ValueError):
        ToolOverlay(mutating=True, removes=True, count_arg="quantity", delta_arg="delta")


def test_removes_on_a_non_mutating_tool_is_a_config_error() -> None:
    """The guard only sees mutating calls; the flag would never fire."""
    with pytest.raises(ValueError):
        ToolOverlay(removes=True)


def test_the_overlay_carries_the_removal_flags() -> None:
    entry = ServerEntry(
        id="dunnes",
        command="x",
        tool_overlays={"adjust": ToolOverlay(mutating=True, removes=True, delta_arg="delta")},
    )
    spec = entry.apply_flags(ToolSpec(server="dunnes", name="adjust", description="a", parameters={}))
    assert (spec.removes, spec.count_arg, spec.delta_arg) == (True, None, "delta")


async def test_an_add_request_does_not_remove(tmp_path: Path) -> None:
    """The 11-09-2026 turn 2: remove refused, so the ledger is not cleared and
    the re-add is answered "already done" -- true, and nothing left the cart."""
    add = _RecordingTool(_add_spec())
    remove = _RecordingTool(_remove_spec(), MCPCallResult(ok=True, content={"removed": 1}))
    mcp = MCPRegistry()
    mcp.register(add)
    mcp.register(remove)
    llm = _TurnScriptedLLM(
        [
            [_add("tomatoes", "c1")],
            [_call("remove_from_cart", {"productId": "1"}, "c2"), _add("tomatoes", "c3")],
        ]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add tomatoes to the cart")
        await _say(h, llm, "add tomatoes to the cart")

    assert remove.calls == []
    assert len(add.calls) == 1
    tool_messages = [m.content or "" for m in llm.passes[-1] if m.role == "tool"]
    refused = next(i for i, m in enumerate(tool_messages) if "not_removed" in m)
    answered = next(i for i, m in enumerate(tool_messages) if "already_done" in m)
    assert refused < answered
    assert "<external>" not in tool_messages[refused]
    assert "removal_refused" in [e.get("event") for e in trace_events(tmp_path)]


def _adjust_spec() -> ToolSpec:
    return _conditional_spec("adjust_cart_quantity_by_name", delta_arg="delta")


def _set_spec() -> ToolSpec:
    return _conditional_spec("set_cart_quantity_by_name", count_arg="quantity")


def _conditional_spec(name: str, *, count_arg: str | None = None, delta_arg: str | None = None) -> ToolSpec:
    param = count_arg or delta_arg
    return ToolSpec(
        server="dunnes",
        name=name,
        description=name,
        parameters={"type": "object", "properties": {"name": {"type": "string"}, param: {"type": "number"}}},
        mutating=True,
        removes=True,
        count_arg=count_arg,
        delta_arg=delta_arg,
    )


async def _one_call(
    tmp_path: Path, spec: ToolSpec, args: dict, utterance: str = "add one more milk"
) -> tuple[list[dict], str]:
    tool = _RecordingTool(spec)
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_call(spec.name, args)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, utterance)
    return tool.calls, _last_tool_message(llm)


@pytest.mark.parametrize("delta, sent", [(1, True), (0.5, True), (-1, False), (0, False), ("down", False)])
async def test_adjust_is_judged_by_its_delta(tmp_path: Path, delta: object, sent: bool) -> None:
    calls, _ = await _one_call(tmp_path, _adjust_spec(), {"name": "milk", "delta": delta})
    assert (len(calls) == 1) is sent


async def test_a_count_free_adjust_is_not_a_guess(tmp_path: Path) -> None:
    """A delta of one is "add one"; only an absolute count is invented."""
    calls, _ = await _one_call(tmp_path, _adjust_spec(), {"name": "milk", "delta": 1}, "add milk")
    assert len(calls) == 1


async def test_a_counted_add_licenses_an_upward_set(tmp_path: Path) -> None:
    calls, _ = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 2})
    assert len(calls) == 1


@pytest.mark.parametrize("utterance", ["add one more milk", "add milk"])
async def test_a_set_to_zero_under_an_add_is_a_removal_first(tmp_path: Path, utterance: str) -> None:
    """The removal check runs ahead of the count check, so the note says what
    is actually wrong: nothing was to come out."""
    calls, message = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 0}, utterance)
    assert calls == []
    assert "not_removed" in message


@pytest.mark.parametrize("quantity", [1, 2])
async def test_a_set_answering_a_count_free_add_is_a_guess(tmp_path: Path, quantity: int) -> None:
    """set(milk, 1) from three takes two out; "add milk" named no number."""
    calls, message = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": quantity}, "add milk")
    assert calls == []
    assert "quantity_needed" in message


async def test_a_counted_add_licenses_a_set(tmp_path: Path) -> None:
    calls, _ = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 2}, "add two milk")
    assert len(calls) == 1


async def test_a_repeat_word_is_not_a_count(tmp_path: Path) -> None:
    """"again" says on top of what is there, not how many: set(milk, 2) from
    three would take one out."""
    calls, message = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 2}, "add milk again")
    assert calls == []
    assert "quantity_needed" in message


def test_count_arg_and_quantity_arg_together_is_a_config_error() -> None:
    with pytest.raises(ValueError):
        ToolOverlay(mutating=True, removes=True, count_arg="quantity", quantity_arg="quantity")


async def test_a_set_to_zero_answering_a_remove_goes_through(tmp_path: Path) -> None:
    """No count in "remove the milk", but it is not an add: nothing is guessed."""
    calls, _ = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 0}, "remove the milk")
    assert len(calls) == 1


async def test_remove_by_volume_is_refused_under_an_add(tmp_path: Path) -> None:
    spec = ToolSpec(
        server="dunnes", name="remove_by_volume", description="v", parameters={"type": "object"},
        mutating=True, removes=True,
    )
    calls, _ = await _one_call(tmp_path, spec, {"name": "milk", "litres": 1})
    assert calls == []


async def test_a_refused_remove_is_refused_on_the_specialist_too(tmp_path: Path) -> None:
    """A remove-only turn meets no goal and escalates; the specialist's
    re-issue goes back through the same guard, so nothing leaves the cart."""

    class _SpecialistRemoves:
        def __init__(self) -> None:
            self.passes = 0

        async def chat(self, messages, tools):
            self.passes += 1
            if self.passes == 1:
                yield _call("remove_from_cart", {"productId": "1"}, "s1")
                return
            yield LLMText(text="Done.")

    remove = _RecordingTool(_remove_spec(), MCPCallResult(ok=True, content={"removed": 1}))
    mcp = MCPRegistry()
    mcp.register(remove)
    llm = _TurnScriptedLLM([[_call("remove_from_cart", {"productId": "1"})]])
    specialist = _SpecialistRemoves()
    async with desk_organizer(
        tmp_path, llm=llm, mcp=mcp, escalate_on_failed=True, specialist_llm=specialist
    ) as h:
        await _say(h, llm, "add tomatoes to the cart")

    assert specialist.passes >= 1
    assert remove.calls == []
    refusals = [e for e in trace_events(tmp_path) if e.get("event") == "removal_refused"]
    assert len(refusals) >= 2


async def test_a_removal_cue_lets_the_remove_through(tmp_path: Path) -> None:
    remove = _RecordingTool(_remove_spec(), MCPCallResult(ok=True, content={"removed": 1}))
    mcp = MCPRegistry()
    mcp.register(remove)
    llm = _TurnScriptedLLM([[_call("remove_from_cart", {"productId": "1"})]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add oat milk instead of the milk")

    assert len(remove.calls) == 1


async def test_the_removal_guard_stands_down_off_english(tmp_path: Path) -> None:
    remove = _RecordingTool(_remove_spec(), MCPCallResult(ok=True, content={"removed": 1}))
    mcp = MCPRegistry()
    mcp.register(remove)
    llm = _TurnScriptedLLM([[_call("remove_from_cart", {"productId": "1"})]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False, reply_language="cs") as h:
        await _say(h, llm, "add rajcata")

    assert len(remove.calls) == 1
