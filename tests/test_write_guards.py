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
from glados.core.utterance import has_quantity_cue, has_repeat_cue, is_add_request, spoken_count
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


@pytest.mark.parametrize("utterance", ["add one more milk", "add milk"])
async def test_a_set_to_zero_under_an_add_is_a_removal_first(tmp_path: Path, utterance: str) -> None:
    """The removal check runs ahead of the count check, so the note says what
    is actually wrong: nothing was to come out."""
    calls, message = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 0}, utterance)
    assert calls == []
    assert "not_removed" in message


@pytest.mark.parametrize(
    "utterance, quantity",
    [
        ("add milk", 1),
        ("add milk", 2),
        ("add milk again", 2),
        ("add two milk", 2),
        ("add some milk", 1),
        ("add more milk", 1),
        ("add one more milk", 4),
    ],
)
async def test_a_set_never_answers_an_add(tmp_path: Path, utterance: str, quantity: int) -> None:
    """"Add" is relative and a set is absolute: set(milk, 2) under "add two
    milk" takes one out of a cart holding three, and the harness cannot see
    the cart to know. A count in the utterance does not bridge that."""
    calls, message = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": quantity}, utterance)
    assert calls == []
    assert "use_add_tool" in message
    assert "<external>" not in message
    assert "set_for_add_refused" in [e.get("event") for e in trace_events(tmp_path)]


@pytest.mark.parametrize(
    "utterance",
    [
        "set the milk to two",
        "buy milk so I have three",
        "add milk up to three in total",
        "put three milk in, exactly three",
    ],
)
async def test_a_set_answering_an_end_state_request_goes_through(tmp_path: Path, utterance: str) -> None:
    """A removal cue, or a count stated as where the cart should end, makes
    the utterance not add-only: the absolute count is the user's to give, and
    the add tool with it would overshoot."""
    calls, _ = await _one_call(tmp_path, _set_spec(), {"name": "milk", "quantity": 3}, utterance)
    assert len(calls) == 1


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


# ---- an identical retry of a call that already failed this turn -------------


_REFUSED_AS_DUPLICATE = MCPCallResult(ok=False, error="Not run: this exact call already completed.")


async def _one_turn_of_identical_adds(tmp_path: Path, attempts: int) -> tuple[_RecordingTool, _TurnScriptedLLM]:
    tool = _RecordingTool(_add_spec(), _REFUSED_AS_DUPLICATE)
    mcp = MCPRegistry()
    mcp.register(tool)
    calls = [_add("milk", f"c{i}") for i in range(attempts)]
    llm = _TurnScriptedLLM([calls])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "now add it back")
    return tool, llm


async def test_a_failed_call_gets_one_identical_retry_and_no_more(tmp_path: Path) -> None:
    """Observed 12-09-2026 (bake-off T8): eight identical adds in one turn, each
    refused by the server as a duplicate. The in-flight set only covers an
    unknown outcome, so every one went back on the wire."""
    tool, llm = await _one_turn_of_identical_adds(tmp_path, attempts=4)

    assert len(tool.calls) == 2
    tool_messages = [m.content or "" for m in llm.passes[-1] if m.role == "tool"]
    assert len(tool_messages) == 4
    assert all("already failed twice in this turn" in m for m in tool_messages[2:])
    assert all("<external>" not in m for m in tool_messages[2:])
    assert [e.get("event") for e in trace_events(tmp_path)].count("retry_refused") == 2


async def test_a_retry_with_different_arguments_is_sent(tmp_path: Path) -> None:
    """Only the IDENTICAL call is pointless; changing it is what the note asks for."""
    tool = _RecordingTool(_add_spec(), _REFUSED_AS_DUPLICATE)
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM(
        [[_add("milk", "c1"), _add("milk", "c2"), _add("low fat milk", "c3")]]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add the milk")

    assert [c["query"] for c in tool.calls] == ["milk", "milk", "low fat milk"]


async def test_the_retry_count_starts_again_next_turn(tmp_path: Path) -> None:
    """A fresh turn is a fresh decision by the user, as for the in-flight set."""
    tool = _RecordingTool(_add_spec(), _REFUSED_AS_DUPLICATE)
    mcp = MCPRegistry()
    mcp.register(tool)
    three = [_add("milk", f"c{i}") for i in range(3)]
    llm = _TurnScriptedLLM([three, three])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add the milk")
        await _say(h, llm, "add the milk again")

    assert len(tool.calls) == 4


async def test_an_unknown_outcome_is_left_to_the_in_flight_set(tmp_path: Path) -> None:
    """An indeterminate result is not a known failure: its re-issue is refused
    as outstanding (never re-sent at all), not counted as a retry."""
    tool = _RecordingTool(_add_spec(), MCPCallResult(ok=False, indeterminate=True, error="timeout"))
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("milk", "c1"), _add("milk", "c2")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add the milk")

    assert len(tool.calls) == 1
    events = [e.get("event") for e in trace_events(tmp_path)]
    assert "reissue_refused" in events and "retry_refused" not in events


class _ScriptedResultsTool(_RecordingTool):
    """Answers each call with the next result in the script."""

    def __init__(self, spec: ToolSpec, results: list[MCPCallResult]) -> None:
        super().__init__(spec)
        self._results = list(results)

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        self.calls.append(dict(args))
        return self._results.pop(0)


async def test_a_success_on_the_same_server_forgets_the_failures(tmp_path: Path) -> None:
    """"Not in the cart" can stop being true once an add lands, so the remove
    that failed twice before it is sent again rather than refused."""
    not_there = MCPCallResult(ok=False, error="not in cart")
    remove = _ScriptedResultsTool(_remove_spec(), [not_there, not_there, not_there])
    add = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(remove)
    mcp.register(add)
    drop = lambda i: _call("remove_from_cart", {"productId": "1"}, f"r{i}")  # noqa: E731
    llm = _TurnScriptedLLM([[drop(0), drop(1), _add("milk", "a1"), drop(2)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "swap the milk for a fresh one")

    assert len(remove.calls) == 3
    assert "retry_refused" not in [e.get("event") for e in trace_events(tmp_path)]


# ---- a count the user said, dropped by the model ----------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("add two more milks to the cart", 2),
        ("Add two more milks to the cart.", 2),
        ("add 3 butters", 3),
        ("add one milk", 1),
        ("put 4 yoghurts in", 4),
        ("please add another two milks", 2),
        ("add milk", None),
        ("add some milk", None),
        ("add more milk", None),
        ("add 2 litres of milk", None),
        ("add two pints of milk", None),
        ("add 500 grams of mince", None),
        ("add 2 carrots and an onion", None),
        ("add 2 carrots, 3 onions", None),
        ("add two milks then bread", None),
        ("add two milks with some bread", None),
        ("remove two milks", None),
        ("set the milk to two", None),
        ("add 7up", None),
        ("add 1L milk", None),
        # A number that is part of the product, found by the code duck: the
        # refusal would have named it and steered the model to multiply.
        ("add a 6 pack of eggs", None),
        ("add a six pack of beer", None),
        ("add 6 pack of eggs", None),
        ("add weetabix 24 pack", None),
        ("add a 12 inch pizza", None),
        ("add Heinz 57 sauce", None),
        ("add number 5 pasta", None),
        ("add 2% milk", None),
        ("add two 2-litre milks", 2),
        ("add 2-litre milk", None),
        ("buy a dozen eggs", None),
        ("add half a dozen eggs", None),
        ("add 2 dozen eggs", None),
        ("add milk for 2 people", None),
        ("add bread by 5", None),
        ("add 2.5kg potatoes", None),
        ("add 1.5 litre coke", None),
    ],
)
def test_spoken_count(text: str, expected: int | None) -> None:
    assert spoken_count(text) == expected


@pytest.mark.parametrize(
    "args, sent",
    [
        ({"query": "milk"}, False),
        ({"query": "milk", "quantity": 1}, False),
        ({"query": "milk", "quantity": 3}, False),
        ({"query": "milk", "quantity": 2}, True),
        ({"query": "milk", "quantity": "2"}, True),
    ],
)
async def test_an_add_must_carry_the_count_the_user_said(tmp_path: Path, args: dict, sent: bool) -> None:
    """Bake-off T10 (12-09-2026): "add two more milks" -> add(milk) with no
    quantity; one carton went in and the turn was reported done."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_call("add_to_cart_by_name", args)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    assert (len(tool.calls) == 1) is sent
    if not sent:
        message = _last_tool_message(llm)
        assert "quantity_mismatch" in message and "quantity 2" in message
        assert "<external>" not in message
        assert "quantity_mismatch_refused" in [e.get("event") for e in trace_events(tmp_path)]


async def test_the_model_can_recover_by_resending_the_count(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM(
        [[_add("milk", "c1", repeat=True), _add("milk", "c2", repeat=True, quantity=2)]]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    assert [c.get("quantity") for c in tool.calls] == [2]


async def test_an_ambiguous_utterance_leaves_the_call_alone(tmp_path: Path) -> None:
    """A list of items: the count cannot be tied to this call, so it stands down."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("onion", "c1")]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add 2 carrots and an onion")

    assert len(tool.calls) == 1


async def test_the_same_call_sent_again_goes_through(tmp_path: Path) -> None:
    """The count is parsed from speech and can be part of the product, so the
    refusal is a nudge, not a wall: the identical call re-sent after it is taken
    as considered. Bounds a wrong parse to the old behaviour, not a loop."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM([[_add("milk", "c1", repeat=True), _add("milk", "c2", repeat=True)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    assert len(tool.calls) == 1
    events = [e.get("event") for e in trace_events(tmp_path)]
    assert events.count("quantity_mismatch_refused") == 1
    assert "quantity_mismatch_overridden" in events


async def test_two_identical_calls_in_one_pass_do_not_override(tmp_path: Path) -> None:
    """Found by the code duck: the second of two identical calls in the SAME
    pass has read no note, so it must be refused too, not waved through."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)

    class _TwoAtOnce(_TurnScriptedLLM):
        async def chat(self, messages, tools):
            self.passes.append([m.model_copy(deep=True) for m in messages])
            if self._pass == 0:
                self._pass = 1
                yield _add("milk", "c1", repeat=True)
                yield _add("milk", "c2", repeat=True)
                return
            yield LLMText(text=self._reply)

    llm = _TwoAtOnce([[]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    assert tool.calls == []


async def test_a_different_wrong_count_is_refused_again(tmp_path: Path) -> None:
    """The override is for the SAME count re-sent. A new wrong answer is judged."""
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    llm = _TurnScriptedLLM(
        [[_add("milk", "c1", repeat=True), _add("milk", "c2", repeat=True, quantity=3)]]
    )
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    assert tool.calls == []


async def test_varying_the_query_cannot_walk_the_turn_to_its_pass_cap(tmp_path: Path) -> None:
    tool = _RecordingTool(_add_spec())
    mcp = MCPRegistry()
    mcp.register(tool)
    queries = ["milk", "milks", "whole milk", "low fat milk"]
    llm = _TurnScriptedLLM([[_add(q, f"c{i}", repeat=True) for i, q in enumerate(queries)]])
    async with desk_organizer(tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False) as h:
        await _say(h, llm, "add two more milks to the cart")

    events = [e.get("event") for e in trace_events(tmp_path)]
    assert events.count("quantity_mismatch_refused") == 2
    assert len(tool.calls) == 2
