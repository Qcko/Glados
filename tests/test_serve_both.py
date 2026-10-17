"""Main + admin servers share one loop: a stop of either stops both, except an
admin viewer that could not bind, which must leave the voice service up."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from glados import _serve_both


class _FakeServer:
    def __init__(self, *, fail_on_start: bool = False, stop_after_s: float | None = None):
        self.should_exit = False
        self.fail_on_start = fail_on_start
        self.stop_after_s = stop_after_s
        self.config = SimpleNamespace(port=9765)
        self.served_until_exit = False

    async def serve(self) -> None:
        if self.fail_on_start:
            raise SystemExit(1)
        waited = 0.0
        while not self.should_exit:
            if self.stop_after_s is not None and waited >= self.stop_after_s:
                return
            await asyncio.sleep(0.01)
            waited += 0.01
        self.served_until_exit = True


def test_admin_bind_failure_keeps_main_serving():
    main = _FakeServer(stop_after_s=0.2)
    admin = _FakeServer(fail_on_start=True)

    asyncio.run(_serve_both(main, admin))

    assert not main.should_exit


def test_admin_stopping_normally_stops_main():
    main = _FakeServer()
    admin = _FakeServer(stop_after_s=0.05)

    asyncio.run(_serve_both(main, admin))

    assert main.should_exit and main.served_until_exit


def test_main_stopping_stops_admin():
    main = _FakeServer(stop_after_s=0.05)
    admin = _FakeServer()

    asyncio.run(_serve_both(main, admin))

    assert admin.should_exit and admin.served_until_exit
