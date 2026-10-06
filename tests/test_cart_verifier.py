"""The I/O half of the cart check (core/cart_verifier.py, DESIGN-cart-verify.md)
and its wiring into the organizer. Each test names the failure it pins; the
numbers match the design's test list."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from glados.core.adapters import LLMMessage, LLMText, LLMToolCall, ToolSpec
from glados.core.cart_verifier import CartVerifier, TurnCart
from glados.mcp.registry import CallEnvelope, MCPCallResult, MCPRegistry
from tests.organizer_harness import CLIENT_ID, desk_organizer, trace_events

MILK = "100806893"
EGGS = "100806924"
SHOP_NAME = "Dunnes Stores Irish Low Fat Milk 3L"
ENVELOPE = CallEnvelope(session_id="s1", room_id="r1", speaker_id="u1")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Cart:
    """A shop cart reachable through a dispatch function, counting reads."""

    def __init__(self, **units: int) -> None:
        self.lines = dict(units)
        self.reads = 0
        self.read_ok = True
        self.during_read = None

    async def dispatch(self, server, name, args, envelope) -> MCPCallResult:
        assert name == "view_cart"
        self.reads += 1
        if self.during_read is not None:
            await self.during_read()
        if not self.read_ok:
            return MCPCallResult(ok=False, error="browser not started")
        lines = [
            {"productId": pid, "name": SHOP_NAME, "quantity": q}
            for pid, q in self.lines.items()
        ]
        return MCPCallResult(ok=True, content={"lines": lines})


def _verifier(cart: _Cart, clock: _Clock | None = None, **kwargs) -> CartVerifier:
    return CartVerifier(
        cart.dispatch, {"dunnes": "view_cart"}, clock=clock or _Clock(), **kwargs
    )


async def _write(verifier: CartVerifier, turn: TurnCart, cart: _Cart, args: dict, **effect: int):
    await verifier.before_write(turn, "dunnes", ENVELOPE)
    verifier.note_write(turn, "dunnes", args)
    cart.lines.update(effect)
    cart.lines = {k: v for k, v in cart.lines.items() if v}
    verifier.note_write_done(turn, "dunnes", MCPCallResult(ok=True))


# ---- CartVerifier -----------------------------------------------------------


async def test_a_turns_own_write_yields_the_diff() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})

    changes = await verifier.after(turn, ENVELOPE)

    assert [(c.product_id, c.before, c.after) for c in changes] == [(MILK, 3, 0)]


async def test_1_another_rooms_write_in_the_window_gives_no_diff() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    verifier.note_write(TurnCart(), "dunnes", {"query": "eggs"})
    cart.lines[EGGS] = 6

    assert await verifier.after(turn, ENVELOPE) is None
    next_turn = TurnCart()
    await verifier.before_write(next_turn, "dunnes", ENVELOPE)
    assert cart.reads == 3


async def test_1_a_write_with_no_turn_still_moves_the_epoch() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    verifier.note_write(None, "dunnes", {"query": "eggs"})

    assert await verifier.after(turn, ENVELOPE) is None


async def test_1_a_write_in_flight_during_the_before_read_is_not_trusted() -> None:
    """Room B's write was sent (epoch moved) but had not landed when room A
    read BEFORE; it lands inside A's window without moving the epoch again."""
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    other = TurnCart()
    verifier.note_write(other, "dunnes", {"query": "eggs"})
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    cart.lines[EGGS] = 6
    verifier.note_write_done(other, "dunnes", MCPCallResult(ok=True))

    assert await verifier.after(turn, ENVELOPE) is None


async def test_a_write_that_raised_makes_the_turn_unsafe() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await verifier.before_write(turn, "dunnes", ENVELOPE)
    verifier.note_write(turn, "dunnes", {"productId": MILK})
    verifier.note_write_done(turn, "dunnes", None)

    assert await verifier.after(turn, ENVELOPE) is None


async def test_2_an_indeterminate_write_skips_the_after_read() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await verifier.before_write(turn, "dunnes", ENVELOPE)
    verifier.note_write(turn, "dunnes", {"productId": MILK})
    verifier.note_write_done(turn, "dunnes", MCPCallResult(ok=False, indeterminate=True))

    assert await verifier.after(turn, ENVELOPE) is None
    assert cart.reads == 1


async def test_3_a_write_during_the_after_read_is_neither_trusted_nor_cached() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 2})

    async def other_room_writes() -> None:
        verifier.note_write(TurnCart(), "dunnes", {"query": "eggs"})

    cart.during_read = other_room_writes
    assert await verifier.after(turn, ENVELOPE) is None
    cart.during_read = None
    await verifier.before_write(TurnCart(), "dunnes", ENVELOPE)
    assert cart.reads == 3


async def test_4_a_failed_after_read_gives_no_diff_and_drops_the_cache() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    cart.read_ok = False

    assert await verifier.after(turn, ENVELOPE) is None


async def test_a_slow_read_times_out_into_no_diff() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart, read_timeout_s=0.05)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    cart.during_read = lambda: asyncio.sleep(1)

    assert await verifier.after(turn, ENVELOPE) is None


async def test_5_barge_in_during_a_read_cancels_it_and_drops_the_cache() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 0})
    started = asyncio.Event()

    async def hang() -> None:
        started.set()
        await asyncio.sleep(10)

    cart.during_read = hang
    task = asyncio.create_task(verifier.after(turn, ENVELOPE))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cart.during_read = None
    await verifier.before_write(TurnCart(), "dunnes", ENVELOPE)
    assert cart.reads == 3


async def test_6_retries_share_one_before_read() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    turn = TurnCart()
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 2})
    await _write(verifier, turn, cart, {"productId": MILK}, **{MILK: 1})

    changes = await verifier.after(turn, ENVELOPE)

    assert cart.reads == 2
    assert [(c.before, c.after) for c in changes] == [(3, 1)]


async def test_7_the_after_read_is_the_next_turns_before() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    first = TurnCart()
    await _write(verifier, first, cart, {"productId": MILK}, **{MILK: 2})
    await verifier.after(first, ENVELOPE)
    second = TurnCart()
    await _write(verifier, second, cart, {"productId": MILK}, **{MILK: 1})

    changes = await verifier.after(second, ENVELOPE)

    assert cart.reads == 3
    assert [(c.before, c.after) for c in changes] == [(2, 1)]


async def test_an_old_cache_is_not_a_before() -> None:
    cart = _Cart(**{MILK: 3})
    clock = _Clock()
    verifier = _verifier(cart, clock, max_age_s=600)
    first = TurnCart()
    await _write(verifier, first, cart, {"productId": MILK}, **{MILK: 2})
    await verifier.after(first, ENVELOPE)
    clock.now += 601

    await verifier.before_write(TurnCart(), "dunnes", ENVELOPE)

    assert cart.reads == 3


async def test_14_a_model_read_overtaken_by_a_write_is_not_cached() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    epoch = verifier.epoch("dunnes")
    verifier.note_write(None, "dunnes", {"query": "eggs"})
    verifier.note_model_read("dunnes", epoch, True, await cart.dispatch("dunnes", "view_cart", {}, ENVELOPE))

    await verifier.before_write(TurnCart(), "dunnes", ENVELOPE)

    assert cart.reads == 2


async def test_a_clean_model_read_saves_the_pre_read() -> None:
    cart = _Cart(**{MILK: 3})
    verifier = _verifier(cart)
    epoch = verifier.epoch("dunnes")
    verifier.note_model_read("dunnes", epoch, True, await cart.dispatch("dunnes", "view_cart", {}, ENVELOPE))

    await verifier.before_write(TurnCart(), "dunnes", ENVELOPE)

    assert cart.reads == 1


def test_attribution_by_product_id_and_by_a_lone_call() -> None:
    verifier = _verifier(_Cart())
    from glados.core.cart_verify import LineChange

    changes = (LineChange(MILK, 0, 1, None), LineChange(EGGS, 0, 1, None))
    lone = TurnCart(write_args=[{"query": "milk"}])
    assert verifier.attribution(lone, changes) == {MILK: "milk", EGGS: "milk"}
    two = TurnCart(write_args=[{"query": "milk"}, {"productId": EGGS, "name": "eggs"}])
    assert verifier.attribution(two, changes) == {EGGS: "eggs"}


# ---- wired into the organizer -----------------------------------------------


class _CartServer:
    """view_cart and remove_from_cart over one in-memory cart."""

    def __init__(self, cart: dict[str, int], remove_ok: bool = True) -> None:
        self.cart = cart
        self.remove_ok = remove_ok

    def tools(self) -> list:
        return [_Tool(self, "view_cart", mutating=False), _Tool(self, "remove_from_cart", mutating=True)]

    def run(self, name: str, args: dict) -> MCPCallResult:
        if name == "view_cart":
            lines = [{"productId": p, "name": SHOP_NAME, "quantity": q} for p, q in self.cart.items()]
            return MCPCallResult(ok=True, content={"lines": lines})
        if not self.remove_ok:
            return MCPCallResult(ok=False, error="Removing the whole line was refused.")
        self.cart.pop(args.get("productId"), None)
        return MCPCallResult(ok=True, content={"text": "Removed."})


class _Tool:
    def __init__(self, server: _CartServer, name: str, *, mutating: bool) -> None:
        self._server = server
        self.spec = ToolSpec(
            server="dunnes", name=name, description=name,
            parameters={"type": "object", "properties": {"productId": {"type": "string"}}},
            mutating=mutating,
        )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        return self._server.run(self.spec.name, args)


class _ScriptedLLM:
    def __init__(self, calls: list[LLMToolCall], reply: str) -> None:
        self._calls = list(calls)
        self._reply = reply
        self.passes: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.passes.append([m.model_copy(deep=True) for m in messages])
        if self._calls:
            yield self._calls.pop(0)
            return
        yield LLMText(text=self._reply)


def _remove(call_id: str = "r1") -> LLMToolCall:
    return LLMToolCall(call_id=call_id, server="dunnes", name="remove_from_cart", args={"productId": MILK})


async def _cart_turn(tmp_path: Path, reply: str, *, remove_ok: bool = True, read_ok: bool = True):
    server = _CartServer({MILK: 3}, remove_ok=remove_ok)
    mcp = MCPRegistry()
    for tool in server.tools():
        mcp.register(tool)

    async def dispatch(srv, name, args, envelope):
        if not read_ok:
            return MCPCallResult(ok=False, error="browser not started")
        return await mcp.dispatch(srv, name, args, envelope)

    verifier = CartVerifier(dispatch, {"dunnes": "view_cart"})
    llm = _ScriptedLLM([_remove()], reply)
    async with desk_organizer(
        tmp_path, llm=llm, mcp=mcp, escalate_on_failed=False, cart_verifier=verifier
    ) as h:
        await h.org.handle_user_text(CLIENT_ID, "Take one of the milks off")
        await h.org.flush()
        llm._reply = "ok"
        await h.org.handle_user_text(CLIENT_ID, "thanks")
        await h.org.flush()
        deltas = [m["text"] for _, m in h.sink if m.get("type") == "assistant_delta"]
        outcomes = [m["outcome"] for _, m in h.sink if m.get("type") == "turn_outcome"]
    history = [m.content for m in llm.passes[-1] if m.role == "assistant" and m.content]
    events = trace_events(tmp_path)
    return deltas, outcomes, history, events


async def test_10_t13_a_removed_line_and_two_remain_is_replaced(tmp_path: Path) -> None:
    lie = "Two 3-litre milks remain."
    deltas, outcomes, history, events = await _cart_turn(tmp_path, lie)

    assert any("(Cart) Took one item out of your cart." in d for d in deltas)
    assert "Took one item out of your cart." in history and lie not in history
    assert outcomes[0] == "done"
    assert "cart_verified" in [e.get("event") for e in events]


async def test_11_a_refused_removal_and_two_remain_says_nothing_changed(tmp_path: Path) -> None:
    lie = "Done, two milks remain."
    _, _, history, _ = await _cart_turn(tmp_path, lie, remove_ok=False)

    assert "Nothing in your cart changed." in history and lie not in history


async def test_12_a_closing_question_survives_a_no_change_line(tmp_path: Path) -> None:
    reply = "I could not remove it. Which milk did you mean?"
    _, _, history, _ = await _cart_turn(tmp_path, reply, remove_ok=False)

    assert "Nothing in your cart changed. Which milk did you mean?" in history


async def test_9_a_failed_read_leaves_the_reply_to_the_existing_chain(tmp_path: Path) -> None:
    reply = "Took the milk out."
    deltas, _, history, events = await _cart_turn(tmp_path, reply, read_ok=False)

    assert reply in history
    assert not any("(Cart)" in d for d in deltas)
    assert "cart_verify_skipped" in [e.get("event") for e in events]


async def test_17_the_harness_reads_never_reach_the_model_or_traces(tmp_path: Path) -> None:
    _, _, _, events = await _cart_turn(tmp_path, "Two remain.")

    traced = repr([e for e in events if e.get("event", "").startswith("cart_")])
    assert SHOP_NAME not in traced
    tool_results = [e for e in events if e.get("event") == "tool_result"]
    assert len(tool_results) == 1
