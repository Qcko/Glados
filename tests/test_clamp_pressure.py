"""B5 -- the clamp alarm, moved off occurrence and onto a streak.

`external_result_capped` fired every time a result was clamped, which on an
ordinary Dunnes page is every turn. These pin the split that fixes it: the
TRACE stays per-occurrence, because an operator reconstructing a turn needs
every clamp in it, while the ALERT waits for a run -- the shape of a tool
returning capped-to-the-limit results pass after pass.
"""

from __future__ import annotations

from pathlib import Path

from glados.core.adapters import ToolSpec
from glados.core.organizer import Organizer
from glados.core.prompt_pressure import DEFAULT_STREAK
from glados.core.sessions import SessionRegistry
from glados.core.tool_payload_cap import clamp_result_bytes
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry

SPEC = ToolSpec(
    server="dunnes",
    name="scan_favorites_for_sales",
    description="scan",
    parameters={"type": "object"},
    untrusted=True,
)


class _Call:
    call_id = "c1"
    server = "dunnes"
    name = "scan_favorites_for_sales"


class _Trace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, name: str, **kw) -> None:
        self.events.append((name, kw))

    def kinds(self) -> list[str]:
        return [name for name, _ in self.events]


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


def _clamp(org: Organizer, trace: _Trace, raw: str) -> None:
    org._note_clamp(clamp_result_bytes(raw, org._max_result_bytes), _Call(), SPEC, trace)


def test_one_long_page_traces_but_does_not_alarm(tmp_path: Path) -> None:
    """The Dunnes case. It is worth recording and not worth waking anyone for,
    and conflating the two is what taught operators to ignore the event."""
    org = _org(tmp_path, max_result_bytes=100)
    trace = _Trace()
    _clamp(org, trace, "x" * 5000)
    assert trace.kinds() == ["external_result_capped"]


def test_sustained_clamping_raises_the_alarm(tmp_path: Path) -> None:
    org = _org(tmp_path, max_result_bytes=100)
    trace = _Trace()
    for _ in range(DEFAULT_STREAK):
        _clamp(org, trace, "x" * 5000)
    assert trace.kinds().count("external_result_capped") == DEFAULT_STREAK
    assert trace.kinds().count("external_clamp_pressure") == 1


def test_a_result_that_fits_breaks_the_streak(tmp_path: Path) -> None:
    """Consecutive is the claim being made. Big pages scattered through a
    session are a busy workload, not a flooding payload."""
    org = _org(tmp_path, max_result_bytes=100)
    trace = _Trace()
    for _ in range(DEFAULT_STREAK - 1):
        _clamp(org, trace, "x" * 5000)
    _clamp(org, trace, "short")
    for _ in range(DEFAULT_STREAK - 1):
        _clamp(org, trace, "x" * 5000)
    assert "external_clamp_pressure" not in trace.kinds()


def test_the_alarm_carries_how_far_over_the_ceiling(tmp_path: Path) -> None:
    """A page 1.1x the cap and a payload 50x it are the same event under a
    boolean, and only one of them is an attack shape."""
    org = _org(tmp_path, max_result_bytes=100)
    trace = _Trace()
    for _ in range(DEFAULT_STREAK):
        _clamp(org, trace, "x" * 5000)
    alarm = [kw for name, kw in trace.events if name == "external_clamp_pressure"][0]
    assert alarm["worst_ratio"] > 10
    assert alarm["streak"] == DEFAULT_STREAK
    assert alarm["untrusted"] is True


def test_a_sustained_run_alarms_once(tmp_path: Path) -> None:
    org = _org(tmp_path, max_result_bytes=100)
    trace = _Trace()
    for _ in range(DEFAULT_STREAK * 4):
        _clamp(org, trace, "x" * 5000)
    assert trace.kinds().count("external_clamp_pressure") == 1
