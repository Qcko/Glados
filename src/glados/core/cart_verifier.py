"""Reads the real cart around a turn's writes (DESIGN-cart-verify.md).

`CartVerifier` owns the I/O half of the cart check: the cached snapshot per
cart account, the per-server write epoch, and the two harness reads. The pure
half -- parse, diff, the spoken line -- is `core/cart_verify.py`.

Rooms run in parallel and share one cart, and nothing serialises their calls
to it. So a diff is trusted only when every write dispatched to that server
between BEFORE and AFTER was this turn's own: the epoch counts every write
from every room, and the turn counts its own. The epoch moves when a write is
SENT, so a read is also only trusted when no write was in flight at its start
or its end -- one sent before the read can land inside it without moving the
epoch (code duck, 06-10-2026).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from glados.core.cart_verify import CartSnapshot, LineChange, diff
from glados.mcp.registry import CallEnvelope, MCPCallResult

log = logging.getLogger(__name__)

Dispatch = Callable[[str, str, dict, CallEnvelope], Awaitable[MCPCallResult]]

_WORD_ARGS = ("query", "name")


@dataclass
class TurnCart:
    """One USER turn's cart writes, across every retry drive of that turn."""

    server: str | None = None
    before: CartSnapshot | None = None
    before_epoch: int = 0
    own_writes: int = 0
    write_args: list[dict] = field(default_factory=list)
    # A write whose outcome is unknown, writes to two carts, or no BEFORE:
    # no diff this turn can be trusted.
    unsafe: bool = False

    @property
    def wrote(self) -> bool:
        return self.own_writes > 0


@dataclass(frozen=True)
class _Cached:
    snapshot: CartSnapshot
    epoch: int
    read_at: float


class CartVerifier:
    def __init__(
        self,
        dispatch: Dispatch,
        cart_reads: Mapping[str, str],
        *,
        read_timeout_s: float = 5.0,
        max_age_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
        persona: bool = False,
    ) -> None:
        self.persona = persona
        self._dispatch = dispatch
        self._cart_reads = dict(cart_reads)
        self._read_timeout_s = read_timeout_s
        self._max_age_s = max_age_s
        self._clock = clock
        self._epochs: dict[str, int] = {}
        self._in_flight: dict[str, int] = {}
        self._cache: dict[str, _Cached] = {}

    def serves(self, server: str) -> bool:
        return server in self._cart_reads

    def is_cart_read(self, server: str, name: str) -> bool:
        return self._cart_reads.get(server) == name

    def epoch(self, server: str) -> int:
        return self._epochs.get(server, 0)

    def quiet(self, server: str) -> bool:
        """No write to this cart is in flight, so a read now sees settled state."""
        return self._in_flight.get(server, 0) == 0

    async def before_write(
        self, turn: TurnCart, server: str, envelope: CallEnvelope
    ) -> None:
        """Pin BEFORE ahead of the turn's first write to this cart."""
        if turn.server not in (None, server):
            turn.unsafe = True
        turn.server = server
        if turn.unsafe or turn.before is not None:
            return
        snapshot = self._valid_cached(server) or await self._read(server, envelope, "BEFORE")
        if snapshot is None:
            turn.unsafe = True
            return
        turn.before = snapshot
        turn.before_epoch = self.epoch(server)

    def note_write(self, turn: TurnCart | None, server: str, args: dict) -> None:
        """Every write to a served cart, from any room, moves its epoch and
        stays in flight until `note_write_done`."""
        if not self.serves(server):
            return
        self._epochs[server] = self.epoch(server) + 1
        self._in_flight[server] = self._in_flight.get(server, 0) + 1
        if turn is not None and turn.server == server:
            turn.own_writes += 1
            turn.write_args.append(dict(args))

    def note_write_done(
        self, turn: TurnCart | None, server: str, result: MCPCallResult | None
    ) -> None:
        """The write came back -- or raised, or was cancelled (`result` None),
        in which case whether it landed is unknown."""
        if not self.serves(server):
            return
        self._in_flight[server] = max(0, self._in_flight.get(server, 0) - 1)
        if turn is not None and (result is None or result.indeterminate):
            turn.unsafe = True

    def note_model_read(
        self, server: str, epoch_at_dispatch: int, quiet_at_dispatch: bool,
        result: MCPCallResult,
    ) -> None:
        """A model-issued cart read refreshes the cache from its RAW result,
        unless a write was in flight or sent while it ran."""
        snapshot = CartSnapshot.from_content(result.content) if result.ok else None
        settled = quiet_at_dispatch and self.quiet(server)
        if snapshot is not None and settled and self.epoch(server) == epoch_at_dispatch:
            self._cache[server] = _Cached(snapshot, epoch_at_dispatch, self._clock())

    async def after(
        self, turn: TurnCart, envelope: CallEnvelope
    ) -> tuple[LineChange, ...] | None:
        """The turn's verified change, or None when it cannot be trusted."""
        server = turn.server
        if server is None or not turn.wrote:
            return None
        if turn.unsafe or turn.before is None:
            self._cache.pop(server, None)
            return None
        snapshot = await self._read(server, envelope, "AFTER")
        if snapshot is None:
            return None
        if self.epoch(server) - turn.before_epoch != turn.own_writes:
            log.info("cart AFTER for %s discarded: another turn wrote in the window", server)
            self._cache.pop(server, None)
            return None
        return diff(turn.before, snapshot)

    def attribution(self, turn: TurnCart, changes: tuple[LineChange, ...]) -> dict[str, str]:
        """productId -> the subject word of the call that touched it. A call
        naming a productId claims that line; a lone write claims every line."""
        words: dict[str, str] = {}
        for args in turn.write_args:
            word = _word_of(args)
            product_id = args.get("productId")
            if word is not None and isinstance(product_id, str):
                words[product_id] = word
        if len(turn.write_args) == 1 and (word := _word_of(turn.write_args[0])):
            for change in changes:
                words.setdefault(change.product_id, word)
        return words

    def _valid_cached(self, server: str) -> CartSnapshot | None:
        cached = self._cache.get(server)
        if cached is None:
            return None
        fresh = self._clock() - cached.read_at < self._max_age_s
        if cached.epoch != self.epoch(server) or not fresh:
            self._cache.pop(server, None)
            return None
        return cached.snapshot

    async def _read(
        self, server: str, envelope: CallEnvelope, label: str
    ) -> CartSnapshot | None:
        """One harness cart read, or None unless the cart was settled for the
        whole read: no write in flight at either end, none sent during it."""
        epoch = self.epoch(server)
        quiet_at_start = self.quiet(server)
        tool = self._cart_reads[server]
        started = self._clock()
        try:
            result = await asyncio.wait_for(
                self._dispatch(server, tool, {}, envelope), self._read_timeout_s
            )
        except TimeoutError:
            log.warning(
                "cart %s read for %s timed out after %ss", label, server, self._read_timeout_s
            )
            self._cache.pop(server, None)
            return None
        except asyncio.CancelledError:
            self._cache.pop(server, None)
            raise
        snapshot = CartSnapshot.from_content(result.content) if result.ok else None
        elapsed_ms = int((self._clock() - started) * 1000)
        settled = quiet_at_start and self.quiet(server) and self.epoch(server) == epoch
        if snapshot is None or not settled:
            log.info("cart %s read for %s unusable (ok=%s, %d ms)", label, server, result.ok, elapsed_ms)
            self._cache.pop(server, None)
            return None
        log.info("cart %s read for %s: %d lines, %d ms", label, server, len(snapshot.lines), elapsed_ms)
        self._cache[server] = _Cached(snapshot, epoch, self._clock())
        return snapshot


def _word_of(args: dict) -> str | None:
    for key in _WORD_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None
