"""Per-room FIFO queue (ARCH §3.4).

Same-room utterances are serialised. Cross-room are parallel. Voice
barge-in clears the room's pending queue in addition to cancelling the
active turn; UI Interrupt does not."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from glados.brain.llm.fake import FakeLLM
from glados.core.adapters import LLMText
from glados.core.config import ClientBinding
from glados.core.organizer import Organizer
from glados.core.room_queues import RoomQueueManager
from glados.core.sessions import SessionRegistry
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry


class _GatedLLM:
    """Yields one delta, then blocks on a gate. Test holds the gate to
    keep one turn 'mid-flight' while observing queue behaviour."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.entry_count = 0

    async def chat(self, messages, tools):
        self.entry_count += 1
        yield LLMText(text="working ")
        self.entered.set()
        await self.release.wait()
        yield LLMText(text="done")


@asynccontextmanager
async def _make_organizer(bindings, tmp, llm):
    sink: list[tuple[str, dict]] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append((client_id, msg.model_dump()))

    by_id = {b.client_id: b for b in bindings}
    org = Organizer(
        llm=llm,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=by_id.get,
        clients_in_room=lambda r: [b.client_id for b in bindings if b.room_id == r],
    )
    try:
        yield org, sink
    finally:
        await org.close()


# ---- RoomQueueManager units --------------------------------------------


@pytest.mark.asyncio
async def test_manager_does_not_spawn_workers_without_enqueue() -> None:
    mgr = RoomQueueManager()
    assert mgr._workers == {}
    await mgr.close()


@pytest.mark.asyncio
async def test_manager_serialises_within_a_room() -> None:
    """Two enqueues to the same room run one after the other, not
    overlapping. Verified by an action that records its own (start, end)
    pair and the other action's start time relative to it."""
    mgr = RoomQueueManager()
    events: list[tuple[str, float]] = []
    loop = asyncio.get_event_loop()

    async def make_action(name: str):
        async def action():
            events.append((f"{name}:start", loop.time()))
            await asyncio.sleep(0.05)
            events.append((f"{name}:end", loop.time()))
        return action

    mgr.enqueue("desk", await make_action("A"))
    mgr.enqueue("desk", await make_action("B"))
    await mgr.flush()
    await mgr.close()

    names = [e[0] for e in events]
    assert names == ["A:start", "A:end", "B:start", "B:end"]


@pytest.mark.asyncio
async def test_manager_parallelises_across_rooms() -> None:
    """Different rooms have independent workers and can run concurrently."""
    mgr = RoomQueueManager()
    a_started = asyncio.Event()
    b_started = asyncio.Event()
    release = asyncio.Event()

    async def a_action():
        a_started.set()
        await release.wait()

    async def b_action():
        b_started.set()
        await release.wait()

    mgr.enqueue("desk", a_action)
    mgr.enqueue("kitchen", b_action)
    await asyncio.wait_for(a_started.wait(), timeout=1.0)
    await asyncio.wait_for(b_started.wait(), timeout=1.0)
    # Both running concurrently → release lets both finish.
    release.set()
    await mgr.flush()
    await mgr.close()


@pytest.mark.asyncio
async def test_manager_clear_drops_pending_only() -> None:
    """clear() removes queued items but does not cancel the running one."""
    mgr = RoomQueueManager()
    ran: list[str] = []
    release = asyncio.Event()

    async def running():
        ran.append("running:start")
        await release.wait()
        ran.append("running:end")

    async def queued():
        ran.append("queued")

    mgr.enqueue("desk", running)
    mgr.enqueue("desk", queued)
    # Let the worker pick up the running item.
    await asyncio.sleep(0)
    while not ran:
        await asyncio.sleep(0)

    assert mgr.queue_depth("desk") == 1  # `queued` waiting
    dropped = mgr.clear("desk")
    assert dropped == 1
    release.set()
    await mgr.flush()
    await mgr.close()
    assert "queued" not in ran


@pytest.mark.asyncio
async def test_manager_continues_after_action_crash(caplog) -> None:
    mgr = RoomQueueManager()
    ran: list[str] = []

    async def boom():
        ran.append("boom")
        raise RuntimeError("synthetic")

    async def ok():
        ran.append("ok")

    with caplog.at_level("ERROR"):
        mgr.enqueue("desk", boom)
        mgr.enqueue("desk", ok)
        await mgr.flush()
    await mgr.close()

    assert ran == ["boom", "ok"]
    assert "synthetic" in caplog.text


@pytest.mark.asyncio
async def test_manager_close_cancels_running_action() -> None:
    """close() forwards CancelledError into the running action so workers
    don't get stranded by an action that swallows its own cancel."""
    mgr = RoomQueueManager()
    inside = asyncio.Event()
    finished_naturally = False

    async def long_running():
        nonlocal finished_naturally
        inside.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Action swallows cancel (like _run_user_text). close() must
            # still stop the worker.
            return
        finished_naturally = True

    mgr.enqueue("desk", long_running)
    await asyncio.wait_for(inside.wait(), timeout=1.0)
    await asyncio.wait_for(mgr.close(), timeout=2.0)
    assert not finished_naturally


# ---- Organizer integration ---------------------------------------------


@pytest.mark.asyncio
async def test_same_room_user_text_is_fifo(tmp_path: Path) -> None:
    """Welcome and Done from the first turn arrive before Welcome for the
    second — proves the queue actually serialises."""
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        FakeLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "first")
        await org.handle_user_text("desk-ui", "second")
        await org.flush()

        # Each turn emits welcome + assistant_delta + done. Two turns →
        # six messages in strict order.
        types = [m["type"] for _, m in sink]
        # First turn's done comes before second turn's welcome.
        first_done_idx = types.index("done")
        second_welcome_idx = types.index("welcome", first_done_idx)
        assert first_done_idx < second_welcome_idx


@pytest.mark.asyncio
async def test_cross_room_turns_can_overlap(tmp_path: Path) -> None:
    """Each room has its own worker; the second room's turn starts before
    the first room's turn finishes."""
    llm = _GatedLLM()
    async with _make_organizer(
        [
            ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko"),
            ClientBinding(client_id="kit-ui", room_id="kitchen", role="ui", default_user="anna"),
        ],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "first")
        await llm.entered.wait()
        await org.handle_user_text("kit-ui", "second")
        # Give kitchen's worker a tick to start its LLM call.
        await asyncio.sleep(0.05)
        assert llm.entry_count == 2, "kitchen turn must start before desk releases"

        # Stronger: both rooms must have actually streamed an assistant_delta
        # while desk is still gated. A serial implementation would only have
        # produced desk's delta so far.
        rooms_with_delta = {
            cid for cid, m in sink if m["type"] == "assistant_delta"
        }
        assert rooms_with_delta == {"desk-ui", "kit-ui"}, (
            f"both rooms must stream concurrently, got {rooms_with_delta}"
        )

        llm.release.set()
        await org.flush()


@pytest.mark.asyncio
async def test_voice_barge_in_clears_room_queue(tmp_path: Path) -> None:
    """A voice 'stop' both cancels the active turn AND drops anything
    queued for that room — voice 'stop' means stop everything."""
    llm = _GatedLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm,
    ) as (org, sink):
        # Turn A starts and hangs in the LLM.
        await org.handle_user_text("desk-ui", "first")
        await llm.entered.wait()
        # Turn B queues behind A.
        await org.handle_user_text("desk-ui", "second")
        assert org._queues.queue_depth("desk") == 1

        # Voice barge-in: cancels A and drops B.
        await org.handle_audio_text("desk-ui", "stop")
        # Let cancellation propagate.
        await org.flush()

        # A was cancelled. B never ran — exactly one welcome.
        welcomes = [m for _, m in sink if m["type"] == "welcome"]
        assert len(welcomes) == 1, f"barge-in must drop queued B (got {welcomes})"
        types = [m["type"] for _, m in sink]
        assert "cancelled" in types


@pytest.mark.asyncio
async def test_barge_in_during_action_startup_window(tmp_path: Path) -> None:
    """Race window: between the worker dequeueing an action and the action
    body registering into `_inflight`, voice barge-in could see "no active
    session, empty queue" and fall through to a regular turn — i.e. "stop"
    becomes "say stop". `_active_actions` (worker-set before await) closes
    that window."""

    class SlowStartLLM:
        """First call hangs *before* yielding anything — the action runs
        but the LLM hasn't started streaming, so the test can race the
        barge-in into the pre-registration window."""

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools):
            self.calls += 1
            await asyncio.sleep(3600)
            yield  # never reached  # pragma: no cover

    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        SlowStartLLM(),
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "first")
        # Yield once so the worker dequeues and creates the action task,
        # but the action hasn't yet entered _inflight (it's only stamped
        # right before the LLM call). Without _active_actions, the next
        # barge-in would fall through.
        await asyncio.sleep(0)
        await org.handle_audio_text("desk-ui", "stop")
        await org.flush()

        # The "stop" must NOT have produced a second welcome.
        welcomes = [m for _, m in sink if m["type"] == "welcome"]
        assert len(welcomes) <= 1, (
            f"barge-in in startup window must not spawn a new turn, got "
            f"{[m for _, m in sink]}"
        )


@pytest.mark.asyncio
async def test_ui_interrupt_does_not_clear_queue(tmp_path: Path) -> None:
    """UI Interrupt cancels the active turn but lets queued ones run —
    typed cancellation is finer-grained than voice 'stop'."""
    llm = _GatedLLM()
    async with _make_organizer(
        [ClientBinding(client_id="desk-ui", room_id="desk", role="ui", default_user="qcko")],
        tmp_path,
        llm,
    ) as (org, sink):
        await org.handle_user_text("desk-ui", "first")
        await llm.entered.wait()
        await org.handle_user_text("desk-ui", "second")

        sid = next(m for _, m in sink if m["type"] == "welcome")["session_id"]
        await org.handle_interrupt("desk-ui", sid)
        # Active turn A cancels; B runs next. Release the gate so B can
        # complete (B will also hit the gate but it's already set after
        # A's release; wait — gate is set per-LLM-instance, so once we
        # release once it's set forever).
        llm.release.set()
        await org.flush()

        welcomes = [m for _, m in sink if m["type"] == "welcome"]
        assert len(welcomes) == 2, "UI interrupt must NOT drop queued B"
        types = [m["type"] for _, m in sink]
        assert "cancelled" in types  # A
        assert "done" in types  # B
