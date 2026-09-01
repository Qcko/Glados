"""The boot-time worst-case budget (core/prompt_budget.py).

The arithmetic is pinned here; the token costs it consumes are measured against
the live model at boot, so nothing in these tests needs a daemon.
"""

from __future__ import annotations

import pytest

from glados.core.adapters import LLMMessage
from glados.core.config import LLMConfig
from glados.core.prompt_budget import (
    dense_payload,
    measure_boot_budget,
    verdict_for,
    worst_case_total,
)


def test_the_worst_case_is_the_prefix_plus_retained_bytes_plus_the_reply() -> None:
    """Retained tool bytes, not turns times per-result.

    The turn-count form was the original shape and it was wrong: turn count
    cannot bound a session when the size of each turn is the attacker's to
    choose.
    """
    assert worst_case_total(100, 400, 1000) == 1500


def test_a_prompt_that_fits_is_reported_as_fitting() -> None:
    v = verdict_for(284, 5389, num_ctx=12288, num_predict=4096)
    assert v.fits
    assert v.worst_case_total == 284 + 5389 + 4096


def test_a_prompt_that_does_not_fit_says_which_knob_to_turn() -> None:
    """A refusal nobody can act on is a refusal that gets commented out."""
    v = verdict_for(284, 20_000, num_ctx=12288, num_predict=4096)
    assert not v.fits
    assert "max_history_external_bytes" in v.detail
    assert "num_ctx" in v.detail


def test_the_dense_payload_fills_the_budget_exactly_and_is_ascii() -> None:
    for budget in (0, 1, 17, 1024, 8192):
        payload = dense_payload(budget)
        assert len(payload.encode("utf-8")) == budget
        payload.encode("ascii")


class _PricingLLM:
    """Answers with a token count proportional to the characters it is given."""

    def __init__(self, per_char: float = 1.0, mute: bool = False) -> None:
        self._per_char = per_char
        self._mute = mute
        self.calls: list[list[LLMMessage]] = []

    async def price_prompt(self, messages, tools):
        self.calls.append(messages)
        if self._mute:
            return None
        return int(sum(len(m.content or "") for m in messages) * self._per_char)


@pytest.mark.asyncio
async def test_the_result_cost_is_the_difference_between_the_two_prices() -> None:
    """Priced as loaded-minus-fixed so the system prompt is not counted twice."""
    llm = _PricingLLM(per_char=1.0)
    v = await measure_boot_budget(
        llm,
        system_prompt="s" * 100,
        tools=[],
        max_history_external_bytes=50,
        num_ctx=100_000,
        num_predict=1000,
    )
    assert v is not None
    assert v.fixed_prefix_tokens == 100
    assert v.retained_external_tokens == 50
    # One fixed-prefix price, then one per candidate worst-case payload shape.
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_a_backend_that_cannot_price_returns_no_verdict() -> None:
    """A fake, or an adapter whose window is not its own to know.

    Unknown is not the same as failing, and the caller decides -- so this must
    not be confused with a budget that does not fit.
    """

    class _Mute:
        pass

    assert (
        await measure_boot_budget(
            _Mute(),
            system_prompt="s",
            tools=[],
            max_history_external_bytes=10,
            num_ctx=100,
            num_predict=10,
        )
        is None
    )


@pytest.mark.asyncio
async def test_an_unreachable_daemon_returns_no_verdict() -> None:
    llm = _PricingLLM(mute=True)
    assert (
        await measure_boot_budget(
            llm,
            system_prompt="s",
            tools=[],
            max_history_external_bytes=10,
            num_ctx=100,
            num_predict=10,
        )
        is None
    )


# ---- the config guard behind it ------------------------------------------


def test_ollama_refuses_an_unset_window() -> None:
    """`None` omits num_ctx, and Ollama's own default is small enough to be the
    attack. Not a sensible default here -- a silent disabling."""
    with pytest.raises(ValueError, match="num_ctx"):
        LLMConfig(backend="ollama", num_ctx=None)


def test_an_explicit_window_is_accepted() -> None:
    assert LLMConfig(backend="ollama", num_ctx=12288).num_ctx == 12288


def test_llamacpp_may_leave_the_window_unset() -> None:
    """There the real window is the server's launch `-c`; this field is only an
    expectation, so refusing it would be refusing an honest admission."""
    cfg = LLMConfig(backend="llamacpp", num_ctx=None, text_tool_format="")
    assert cfg.num_ctx is None


# ---- the boot gate -------------------------------------------------------


def test_a_budget_that_does_not_fit_is_reported_as_not_fitting() -> None:
    """What the lifespan turns into a refusal to start.

    Front-truncation is silent, so the alternative to failing here is a model
    that quietly stopped being told to treat external content as data.
    """
    v = verdict_for(4000, 6000, num_ctx=12288, num_predict=4096)
    assert not v.fits
    assert v.worst_case_total == 14096


def test_the_boundary_is_inclusive() -> None:
    """Exactly filling the window is not an overflow."""
    v = verdict_for(1000, 1000, num_ctx=3000, num_predict=1000)
    assert v.fits
    assert v.worst_case_total == 3000
    assert not verdict_for(1001, 1000, num_ctx=3000, num_predict=1000).fits


class _ShapeSensitiveLLM:
    """Prices the two candidate payload shapes differently."""

    def __init__(self, cheap: int, dear: int) -> None:
        self._cheap = cheap
        self._dear = dear
        self._seen = 0

    async def price_prompt(self, messages, tools):
        self._seen += 1
        if self._seen == 1:
            return 0
        return self._cheap if self._seen == 2 else self._dear


@pytest.mark.asyncio
async def test_the_worse_of_the_two_payload_shapes_is_believed() -> None:
    """Neither shape is provably the worst case, so the budget takes the max.

    Measured 01-09-2026: the repeated unit priced 1.14 bytes/token and the
    pseudo-random walk 1.27 -- the opposite of the expectation that repetition
    would compress. Picking by argument was how that got backwards; pricing
    both and believing the worse removes the guess.
    """
    v = await measure_boot_budget(
        _ShapeSensitiveLLM(cheap=10, dear=99),
        system_prompt="s",
        tools=[],
        max_history_external_bytes=64,
        num_ctx=100_000,
        num_predict=10,
    )
    assert v is not None
    assert v.retained_external_tokens == 99


@pytest.mark.asyncio
async def test_a_payload_that_compresses_is_refused() -> None:
    """A pricing payload looser than real hostile content under-prices the
    budget, and would pass a configuration that overflows in production."""

    class _Compressing:
        async def price_prompt(self, messages, tools):
            return 0 if len(messages) == 1 else 2

    with pytest.raises(RuntimeError, match="bytes/token"):
        await measure_boot_budget(
            _Compressing(),
            system_prompt="s",
            tools=[],
            max_history_external_bytes=4096,
            num_ctx=100_000,
            num_predict=10,
        )
