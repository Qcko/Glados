"""Per-room FIFO queue manager.

ARCH section 3.4: "Per-session FIFO so one room's commands stay ordered.
Cross-session: parallel if the LLM backend supports it (vLLM, llama.cpp
parallel slots); fair round-robin if not."

A "room" here is the unit of speaker-output coherence -- two simultaneous
TTS streams to one room are incoherent regardless of who spoke, so the
queue key is `room_id`, not `(room_id, speaker_id)`. Cross-room
parallelism falls out for free: each room owns its own worker task, and
the LLM backend serialises (Ollama) or parallelises (vLLM) on its own
terms -- the manager doesn't impose extra serialisation.

Each action runs as a child task of the worker. Two cancellation paths
matter and they must not be confused:

  - Barge-in cancels the action's task via `_inflight` (Organizer owns
    that map). The action catches `CancelledError`, emits `Cancelled`,
    returns normally. The worker's `await action_task` sees a normal
    completion and moves to the next queued item.
  - `close()` cancels the worker itself. The worker forwards that
    cancel to the active `action_task`, drains it, and re-raises so the
    task ends.

Worker doing `await action()` directly would not work: when the action
swallows its own `CancelledError`, asyncio considers the worker's
cancellation handled, and `close()` silently fails to stop the worker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

Action = Callable[[], Awaitable[None]]


class RoomQueueManager:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Action]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        # Each worker publishes its currently-running action task here so
        # `close()` can cancel it directly (more reliable than cancelling
        # the worker and relying on the worker's own cancel-handler to
        # forward the signal -- see close() for the rationale).
        self._active_actions: dict[str, asyncio.Task] = {}

    def enqueue(
        self, room_id: str, action: Action, *, max_depth: int | None = None
    ) -> bool:
        """Append `action` to `room_id`'s FIFO. Spawns the room's worker
        on first enqueue. Synchronous -- returns once the item is queued,
        not once it completes.

        `max_depth` bounds what the CALLER is willing to pile up behind:
        with it set, an enqueue onto a room already holding that many
        pending actions queues nothing and returns False. Opt-in per call
        rather than a property of the queue, because the two callers want
        opposite answers -- a human's own turn in their own room must never
        be dropped on the floor, while an announcement from elsewhere costs
        that room's occupant head-of-line time and is the thing a bound
        exists to stop accumulating.

        The depth counts everything pending, not announcements alone. A room
        whose own occupant already has a backlog is the room an announcement
        has least claim to join.
        """
        queue = self._queues.get(room_id)
        if queue is not None and max_depth is not None and queue.qsize() >= max_depth:
            log.info(
                "room %s queue at depth %d; refusing a bounded enqueue (max %d)",
                room_id, queue.qsize(), max_depth,
            )
            return False
        if queue is None:
            queue = asyncio.Queue()
            self._queues[room_id] = queue
            self._workers[room_id] = asyncio.create_task(
                self._worker(room_id, queue),
                name=f"room-worker:{room_id}",
            )
        queue.put_nowait(action)
        return True

    def clear(self, room_id: str) -> int:
        """Drop pending actions from a room's queue. Returns the number
        dropped. The currently-running action is unaffected -- use
        `Organizer.handle_interrupt` for that. Called by voice barge-in
        to prevent queued utterances from running after a "stop"."""
        queue = self._queues.get(room_id)
        if queue is None:
            return 0
        n = 0
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            queue.task_done()
            n += 1
        return n

    def queue_depth(self, room_id: str) -> int:
        queue = self._queues.get(room_id)
        return queue.qsize() if queue is not None else 0

    async def flush(self) -> None:
        """Wait until every room's queue is fully drained, including queues
        that did not exist when the call was made.

        A single pass over a snapshot is not enough: a turn in one room can
        hand work to another (the intercom), creating that room's queue and
        its worker mid-flush. A snapshot pass returns while that work is still
        running, and a caller that tears down straight afterwards cancels it --
        which reads as the feature dropping messages rather than as the flush
        being early. So this runs passes until it finds a fixed point: nothing
        queued anywhere and no action still running.

        Test hook -- production callers should not need this because in-flight
        turns broadcast their own Done/Cancelled and observers wait on the
        wire."""
        while True:
            # Yields once per pass so that a pass which finds nothing to await
            # cannot spin: every suspension point below is conditional, and a
            # loop whose only `await` is conditional is a busy-wait waiting to
            # happen.
            await asyncio.sleep(0)
            queues = list(self._queues.values())
            idle = all(queue.empty() for queue in queues) and all(
                task.done() for task in self._active_actions.values()
            )
            if idle:
                return
            for queue in queues:
                await queue.join()

    async def close(self) -> None:
        """Cancel all room workers and the action they're currently running.
        Pending queue items are dropped. Bounded by a per-worker timeout so
        a misbehaving action handler cannot block server shutdown."""
        # Cancel actions first so each worker's `await action_task` resolves
        # via a completed (cancelled) child rather than via its own cancel --
        # avoids the Python 3.11+ "double cancel" path where a worker that
        # catches its own CancelledError finds every subsequent await
        # raising CancelledError again before action_task can finish its
        # finally-block cleanup (Cancelled broadcast, trace.close).
        for room_id in list(self._workers):
            await self._cancel_active_action(room_id)
        for task in list(self._workers.values()):
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._queues.clear()
        self._workers.clear()
        # Cleared here too, not only by the worker's finally: both waits above
        # are bounded, and a timeout leaves a not-done task in this map with
        # the queue dict already emptied -- the one state in which `flush`'s
        # pass finds nothing to await and spins on a live core.
        self._active_actions.clear()

    async def _cancel_active_action(self, room_id: str) -> None:
        """If the worker for `room_id` is mid-action, cancel that action and
        wait for it to finish. The action's own handler (today:
        `_run_user_text`) catches CancelledError and runs its finally --
        including the shielded Cancelled broadcast -- before returning."""
        active = self._active_actions.get(room_id)
        if active is None or active.done():
            return
        active.cancel()
        try:
            await asyncio.wait_for(active, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

    async def _worker(self, room_id: str, queue: asyncio.Queue[Action]) -> None:
        while True:
            action = await queue.get()
            action_task = asyncio.create_task(
                action(), name=f"turn:{room_id}"
            )
            self._active_actions[room_id] = action_task
            try:
                await action_task
            except asyncio.CancelledError:
                # Two paths reach here:
                #   1. close() cancelled action_task first, then us. The
                #      child finished normally (its own handler ran);
                #      `await action_task` raises CancelledError because
                #      WE are still flagged cancelled. Re-raise to end
                #      the worker.
                #   2. Some upstream cancelled the worker directly. Forward
                #      to the action, but don't block on it -- its cleanup
                #      will run on the loop after we exit.
                if not action_task.done():
                    action_task.cancel()
                raise
            except Exception:
                log.exception("turn handler crashed in room %s", room_id)
            finally:
                self._active_actions.pop(room_id, None)
                queue.task_done()
