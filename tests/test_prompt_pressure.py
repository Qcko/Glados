"""Unit cover for B5's runtime half.

The adapter tests exercise this through a real stream; these pin the judgement
itself, where the boundaries are cheap to state and a retuned constant has
nowhere to hide.
"""

from __future__ import annotations

import pytest

from glados.core.adapters import LLMMessage
from glados.core.prompt_budget import MAX_CREDIBLE_BYTES_PER_TOKEN
from glados.core.prompt_pressure import (
    DEFAULT_STREAK,
    ESCALATION_FACTOR,
    PromptPressureMonitor,
    StreakAlarm,
    estimated_prompt_tokens,
)


def _fire(alarm: StreakAlarm, ratio: float, times: int):
    fired = [alarm.observe(ratio, "detail") for _ in range(times)]
    return [f for f in fired if f is not None]


def test_a_single_breach_is_silent() -> None:
    """The whole change. One big page is not a pattern and must not speak."""
    assert _fire(StreakAlarm("k"), 2.0, DEFAULT_STREAK - 1) == []


def test_a_run_of_breaches_speaks_once() -> None:
    assert len(_fire(StreakAlarm("k"), 2.0, DEFAULT_STREAK * 4)) == 1


def test_the_alarm_carries_the_worst_ratio_not_the_last() -> None:
    """One line reaches the operator, so it has to be the worst line."""
    alarm = StreakAlarm("k")
    alarm.observe(9.0, "d")
    alarm.observe(1.5, "d")
    fired = alarm.observe(1.2, "d")
    assert fired is not None
    assert fired.ratio == 9.0


def test_a_clean_observation_breaks_the_streak() -> None:
    """Consecutive is the claim. Unrelated breaches spread across a session are
    a busy workload, not the flooding shape, and counting them together would
    reintroduce the noise the streak exists to filter."""
    alarm = StreakAlarm("k")
    for _ in range(DEFAULT_STREAK - 1):
        alarm.observe(2.0, "d")
    alarm.observe(0.5, "d")
    assert _fire(alarm, 2.0, DEFAULT_STREAK - 1) == []


def test_exactly_at_the_threshold_is_not_a_breach() -> None:
    assert _fire(StreakAlarm("k"), 1.0, DEFAULT_STREAK * 2) == []


def test_re_arms_after_the_condition_clears() -> None:
    alarm = StreakAlarm("k")
    _fire(alarm, 2.0, DEFAULT_STREAK)
    alarm.observe(0.5, "d")
    assert len(_fire(alarm, 2.0, DEFAULT_STREAK)) == 1


def test_estimate_ignores_the_first_system_message() -> None:
    """The system prompt is inside `fixed_prefix_tokens`, priced at boot through
    the real model. Counting its bytes here would double-count it."""
    msgs = [
        LLMMessage(role="system", content="x" * 1000),
        LLMMessage(role="user", content="y" * 16),
    ]
    assert estimated_prompt_tokens(
        msgs, fixed_prefix_tokens=500, bytes_per_token=1.6
    ) == 510


def test_estimate_counts_a_later_system_message() -> None:
    """Only the FIRST system message is inside the boot price. A harness
    directive riding with the turn (`_finish_the_job`'s nudge) is really sent
    and really costs tokens, and skipping it would leave those tokens paid for
    by nothing -- pushing `actual > estimate` for a reason that says nothing
    about density."""
    msgs = [
        LLMMessage(role="system", content="x" * 1000),
        LLMMessage(role="system", content="z" * 160),
    ]
    assert estimated_prompt_tokens(
        msgs, fixed_prefix_tokens=500, bytes_per_token=1.6
    ) == 600


def test_estimate_counts_tool_call_arguments() -> None:
    """The adapters serialize the argument dict onto the wire, so those bytes
    are in the prompt whether or not they are in `content`."""
    from glados.core.adapters import LLMToolCall

    call = LLMToolCall(call_id="c1", server="s", name="n", args={"q": "y" * 100})
    bare = estimated_prompt_tokens(
        [LLMMessage(role="assistant", content=None)],
        fixed_prefix_tokens=0,
        bytes_per_token=1.6,
    )
    withcall = estimated_prompt_tokens(
        [LLMMessage(role="assistant", content=None, tool_calls=[call])],
        fixed_prefix_tokens=0,
        bytes_per_token=1.6,
    )
    assert bare == 0
    assert withcall > 60


def test_estimate_prices_the_tool_schema_block_from_boot_only() -> None:
    """The schema block is not made of message bytes, so it cannot be estimated
    from characters -- that would be estimating something that is not there."""
    assert estimated_prompt_tokens(
        [], fixed_prefix_tokens=3300, bytes_per_token=1.6
    ) == 3300


def test_drift_needs_an_adopted_boot_budget() -> None:
    mon = PromptPressureMonitor(num_ctx=None, num_predict=None)
    assert mon.estimate_for([LLMMessage(role="user", content="hi")]) is None
    for _ in range(DEFAULT_STREAK * 2):
        assert mon.observe(99_999) == []


def test_drift_fires_when_actual_exceeds_the_assumed_density() -> None:
    mon = PromptPressureMonitor(num_ctx=None, num_predict=None)
    mon.adopt_boot_budget(100)
    kinds = [
        a.kind
        for _ in range(DEFAULT_STREAK)
        for a in mon.observe(5000, estimated_tokens=200)
    ]
    assert kinds == ["estimator_drift"]


def test_prose_prices_under_the_estimate() -> None:
    """Sanity on the direction that makes this usable rather than noise: at the
    measured 4.49 bytes/token for English, a prose prompt costs well under an
    estimate built at 1.6."""
    prose = "the quick brown fox jumps over the lazy dog. " * 40
    msgs = [LLMMessage(role="user", content=prose)]
    estimate = estimated_prompt_tokens(
        msgs, fixed_prefix_tokens=0, bytes_per_token=MAX_CREDIBLE_BYTES_PER_TOKEN
    )
    assert len(prose.encode("utf-8")) / 4.49 < estimate


def test_coupling_is_judged_against_the_usable_window() -> None:
    """The band the design flagged: 9000 tokens is under 0.8 * 12288 (9830) so
    the old check was silent, while 9000 + 4096 already exceeds the window."""
    mon = PromptPressureMonitor(num_ctx=12288, num_predict=4096)
    kinds = [a.kind for _ in range(DEFAULT_STREAK) for a in mon.observe(9000)]
    assert kinds == ["context_pressure"]


def test_coupling_boundary_is_the_usable_window_with_no_fudge() -> None:
    """No fraction on top of the reservation, and the reason is load-bearing.

    Applying the old 0.8 here double-counted the margin: it breached at 6553
    while the boot check certifies a worst case near 7500 and passes it, so the
    certified configuration would alarm on every send. The invariant breaches at
    `usable` exactly, and that is still early -- truncation is a further
    num_predict tokens away."""
    usable = 12288 - 4096
    under = PromptPressureMonitor(num_ctx=12288, num_predict=4096)
    over = PromptPressureMonitor(num_ctx=12288, num_predict=4096)
    for _ in range(DEFAULT_STREAK):
        assert under.observe(usable) == []
    fired = [a for _ in range(DEFAULT_STREAK) for a in over.observe(usable + 1)]
    assert [a.kind for a in fired] == ["context_pressure"]


def test_the_certified_worst_case_does_not_alarm() -> None:
    """The regression this replaced. `prompt_budget` blesses ~7500 prompt tokens
    at the shipped config; the alarm that fires on the one shape the design is
    built around is an alarm nobody can leave switched on."""
    mon = PromptPressureMonitor(num_ctx=12288, num_predict=4096)
    for _ in range(DEFAULT_STREAK * 2):
        assert mon.observe(7508) == []


def test_a_standing_breach_speaks_again_when_it_gets_materially_worse() -> None:
    """A lowered num_ctx or a grown tool block never produces a clean
    observation, so without this the gate is the old one-shot latch with a delay
    in front of it -- silent while the ratio climbs from 1.05x to 5x."""
    alarm = StreakAlarm("k")
    first = _fire(alarm, 1.1, DEFAULT_STREAK)
    assert len(first) == 1
    assert _fire(alarm, 1.1 * (ESCALATION_FACTOR - 0.05), 3) == []
    worse = _fire(alarm, 1.1 * ESCALATION_FACTOR * 1.1, 1)
    assert len(worse) == 1
    assert worse[0].ratio > first[0].ratio


def test_unknown_window_silences_coupling_but_not_drift() -> None:
    """llama.cpp's real window is llama-server's launch `-c`, which the adapter
    cannot see. Guessing would be worse than silence -- but drift needs no
    window and must keep working."""
    mon = PromptPressureMonitor(num_ctx=None, num_predict=4096)
    mon.adopt_boot_budget(10)
    kinds = [
        a.kind
        for _ in range(DEFAULT_STREAK)
        for a in mon.observe(9999, estimated_tokens=20)
    ]
    assert kinds == ["estimator_drift"]


def test_a_reservation_wider_than_the_window_silences_coupling() -> None:
    """Misconfiguration, not an attack. There is no usable window to judge
    against, and a negative one would make every prompt a breach -- the boot
    check is where that configuration should be refused."""
    mon = PromptPressureMonitor(num_ctx=1024, num_predict=4096)
    for _ in range(DEFAULT_STREAK * 2):
        assert mon.observe(500) == []


def test_no_token_count_is_not_a_breach() -> None:
    """A response without usage says nothing about pressure. Treating a missing
    number as zero, or as an overflow, both invent a fact."""
    mon = PromptPressureMonitor(num_ctx=8192, num_predict=4096)
    for _ in range(DEFAULT_STREAK * 2):
        assert mon.observe(None) == []


@pytest.mark.parametrize("streak", [1, 2, 5])
def test_streak_length_is_configurable(streak: int) -> None:
    mon = PromptPressureMonitor(num_ctx=8192, num_predict=0, streak=streak)
    fired = [a for _ in range(streak) for a in mon.observe(9000)]
    assert len(fired) == 1
