"""Runtime half of B5: notice when the boot guarantee stops being true.

`prompt_budget` proves an inequality once, at boot, against numbers priced
through the real tokenizer. Everything downstream of that is a promise that the
priced numbers still describe the prompts actually being sent. This module is
where that promise is checked, and it is a module rather than another block
inside each adapter because the judgement is one judgement: the adapters differ
in where they read a token count from, not in what an over-budget prompt means.

Three things are watched, and all three are surfaced the same way -- see
`StreakAlarm` for why that shape and not a per-occurrence warning.

**Drift.** The boot check priced the retained-bytes ceiling as token-dense
content and asserted it costs no more than `_MAX_CREDIBLE_BYTES_PER_TOKEN`. If
a live prompt costs MORE tokens than that density predicts for the bytes it
carried, the boot pricing was optimistic and the inequality it proved does not
cover the prompts being sent. That is the one failure that makes every other
control here a formality, and nothing was watching for it.

**Coupling.** Both adapters tested `prompt_tokens > 0.8 * num_ctx` while the
invariant that matters is `prompt_tokens + num_predict <= num_ctx`. At shipped
values those disagree over a 1638-token band (9830 against 8192) in which the
reply reservation is already blown and nothing said so. The ratio here is taken
against the USABLE window -- `num_ctx` less `num_predict` -- so the warning
arrives before that band rather than inside it.

**Clamp pressure.** `external_result_capped` fires every time a result is
clamped, which on an ordinary Dunnes page is every turn. An operator who sees
it that often stops seeing it, and the signal that matters -- a tool returning
capped-to-the-limit results pass after pass, which is the flooding shape -- is
buried in the noise it shares a name with.

Nothing here decides policy or changes a prompt. It reports; the caller logs,
traces, or ignores. A monitor that could shed would be a second shedding path
disagreeing with `Organizer._shed_for_hop`, and the ceiling that was measured
is the one that should bind.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .adapters import LLMMessage
from .prompt_budget import MAX_CREDIBLE_BYTES_PER_TOKEN

# Consecutive breaches before an alarm speaks. One breach is a big page; a run
# of them is a pattern. Three is the smallest number that is not "occurrence"
# while still firing inside a single turn's tool loop, which runs up to eight
# passes -- an alarm that needed a whole session would miss the shape it is for.
DEFAULT_STREAK = 3

# How much worse a standing breach has to get before it is worth a second line.
# Without this the streak gate is the old one-shot latch with a delay in front
# of it: a condition that never clears never produces a clean observation, so it
# would warn once per process and stay silent while going from 1.05x to 5x.
ESCALATION_FACTOR = 1.25


@dataclass(frozen=True)
class Alarm:
    """A breach that has persisted long enough to be worth saying out loud."""

    kind: str
    streak: int
    ratio: float
    detail: str


def log_alarms(
    log: logging.Logger, alarms: list[Alarm], *, source: str
) -> None:
    """Emit alarms identically from every call site.

    The wording lives here rather than at each site because the divergent
    warning strings in `ollama.py` and `llamacpp.py` were the visible half of
    the duplicated judgement this module replaces: one condition read as two
    different problems depending on which backend happened to be running. The
    organizer's clamp alarm shares it for the same reason, which is why this
    says "observations" rather than "sends" -- the unit differs by caller and
    the sentence should not.
    """
    for alarm in alarms:
        log.warning(
            "%s: %s. Sustained over %d consecutive observations, worst %.2fx "
            "the budget, source %s.",
            alarm.kind,
            alarm.detail,
            alarm.streak,
            alarm.ratio,
            source,
        )


class StreakAlarm:
    """Fire on a RUN of breaches, carrying how bad the worst of them was.

    The alternative -- warn on every occurrence -- is what shipped, and the B5
    table already names its failure: an alert that fires on every Dunnes page
    is muted within a week, and a muted alert is worse than none because it
    reads as coverage. So a single breach is counted and stays silent.

    A sustained breach produces one alarm rather than one per send, and `ratio`
    is the PEAK across the streak rather than the latest value: the operator
    gets one line, so it has to be the worst line.

    But "one line" cannot mean "one line ever", and that is the trap this class
    walked into on its first draft. The conditions worth alarming on -- a
    lowered `num_ctx`, a tool block that grew -- are STANDING conditions, which
    by definition never produce the clean observation that re-arms the gate. So
    the first version warned once per process and then watched the ratio climb
    from 1.05x to 5x in silence: the one-shot latch it replaced, with a delay
    bolted on. `ESCALATION_FACTOR` is what makes it a gate rather than a latch.

    Not thread-safe, and deliberately not: one monitor is shared by every room
    an adapter serves, so "consecutive" means consecutive SENDS, not consecutive
    sends within one conversation. A breach in one room can be reset by a clean
    send in another. That is a weaker claim than it looks, and it is the right
    trade only because the conditions being watched are properties of the
    process (window, tool block, boot pricing) rather than of a session.
    """

    def __init__(self, kind: str, *, streak: int = DEFAULT_STREAK) -> None:
        self._kind = kind
        self._streak_required = max(1, streak)
        self._streak = 0
        self._peak = 0.0
        self._fired_at = 0.0

    def reset(self) -> None:
        """Forget the current run. A clean observation, said explicitly."""
        self._streak = 0
        self._peak = 0.0
        self._fired_at = 0.0

    def observe(self, ratio: float, detail: str) -> Alarm | None:
        """`ratio` is the breach magnitude: at or below 1.0 is not a breach."""
        if ratio <= 1.0:
            self.reset()
            return None
        self._streak += 1
        self._peak = max(self._peak, ratio)
        if self._streak < self._streak_required:
            return None
        if self._fired_at and self._peak < self._fired_at * ESCALATION_FACTOR:
            return None
        self._fired_at = self._peak
        return Alarm(
            kind=self._kind,
            streak=self._streak,
            ratio=self._peak,
            detail=detail,
        )


def estimated_prompt_tokens(
    messages: list[LLMMessage],
    *,
    fixed_prefix_tokens: int,
    bytes_per_token: float,
) -> int:
    """What the boot check's own assumptions predict this prompt costs.

    Deliberately NOT a second tokenizer. Two estimators that are supposed to
    agree are two estimators that will drift, and that drift would surface as a
    disagreement nobody could adjudicate. This one is built from exactly the two
    quantities the boot check already measured, so a gap between it and reality
    is a statement about the boot check rather than about itself.

    `fixed_prefix_tokens` is the system prompt plus the tool schema block,
    priced at boot through the real model because the schema block is not made
    of message bytes -- estimating it from characters would be estimating
    something that is not there. Everything else is message text, converted at
    the densest bytes-per-token ratio the boot check is willing to believe.

    The direction is what makes this usable. English prose measured 4.49
    bytes/token, so ordinary conversation prices far UNDER this estimate and
    stays quiet. Only content denser than the boot check's assumed worst case
    can exceed it -- which is precisely the condition that invalidates the boot
    check, and precisely the shape of a flooding payload.

    Asking the model instead (`price_prompt` with `num_predict=0`) is exact and
    is what boot does, but it is a second full prompt evaluation on every send:
    a per-turn cost paid to detect a condition that should never occur.

    Only the FIRST system message is excluded, because only that one is inside
    `fixed_prefix_tokens`. A harness directive riding with the turn (today
    `_finish_the_job`'s nudge) is a later system message that really is sent and
    really does cost tokens, and skipping every system message would have left
    it paid for by nothing. `tool_calls` are counted for the same reason: the
    adapters serialize the argument dict onto the wire, so those bytes are in
    the prompt whether or not they are in `content`.

    Both omissions pushed the same way -- estimate too low, so `actual >
    estimate` for reasons that say nothing about density. An alarm made noisy by
    its own arithmetic is the failure mode the streak gate exists to prevent,
    and it would have been the harder one to diagnose.
    """
    body_bytes = 0
    seen_system = False
    for m in messages:
        if m.role == "system" and not seen_system:
            seen_system = True
            continue
        if m.content:
            body_bytes += len(m.content.encode("utf-8"))
        for call in m.tool_calls or ():
            body_bytes += len(json.dumps(call.args).encode("utf-8"))
    return fixed_prefix_tokens + int(body_bytes / bytes_per_token)


class PromptPressureMonitor:
    """One adapter's view of whether the boot guarantee still holds.

    Adapters supply numbers -- a token count read from their own response
    shape -- and this decides what the numbers mean. `num_ctx` may be None
    (llama.cpp's real window is `llama-server`'s launch `-c`, which the adapter
    cannot see): the coupling check then goes quiet rather than guessing, while
    drift, which needs no window at all, keeps working.
    """

    def __init__(
        self,
        *,
        num_ctx: int | None,
        num_predict: int | None,
        streak: int = DEFAULT_STREAK,
    ) -> None:
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._coupling = StreakAlarm("context_pressure", streak=streak)
        self._drift = StreakAlarm("estimator_drift", streak=streak)
        self._fixed_prefix_tokens: int | None = None
        self._bytes_per_token = MAX_CREDIBLE_BYTES_PER_TOKEN

    def adopt_boot_budget(
        self, fixed_prefix_tokens: int, *, bytes_per_token: float | None = None
    ) -> None:
        """Take the prices the boot check measured.

        Until this is called `estimate_for` returns None and the drift check
        stays silent, which is the correct posture rather than a gap: without a
        real price for the tool schema block there is no estimate to compare
        against, and inventing one would manufacture the false alarms this
        module exists to avoid. The boot check already declines to run when the
        backend cannot price a prompt, so the two go quiet together.

        `bytes_per_token` is the density the boot check ACHIEVED, and it
        matters that it is not the ceiling the check asserts against.
        `MAX_CREDIBLE_BYTES_PER_TOKEN` is 1.6 because that is the loosest
        density the budget will tolerate before refusing to boot; the density
        actually measured on the shipped model was 1.14. Estimating at 1.6
        predicts roughly 40% fewer tokens than the boot check's own measurement
        of the same kind of content, so legitimately dense bytes -- base64,
        minified JSON, source code, any multibyte script -- would price above
        the estimate without the boot check having been wrong about anything.
        The drift alarm is supposed to be a statement ABOUT the boot check, so
        it has to be calibrated against what that check measured. The ceiling
        remains the fallback for a backend that never reports one.
        """
        self._fixed_prefix_tokens = fixed_prefix_tokens
        if bytes_per_token and bytes_per_token > 0:
            self._bytes_per_token = bytes_per_token

    def estimate_for(self, messages: list[LLMMessage]) -> int | None:
        if self._fixed_prefix_tokens is None:
            return None
        return estimated_prompt_tokens(
            messages,
            fixed_prefix_tokens=self._fixed_prefix_tokens,
            bytes_per_token=self._bytes_per_token,
        )

    def observe(
        self, prompt_tokens: int | None, *, estimated_tokens: int | None = None
    ) -> list[Alarm]:
        if prompt_tokens is None:
            return []
        raised = [
            self._observe_coupling(prompt_tokens),
            self._observe_drift(prompt_tokens, estimated_tokens),
        ]
        return [alarm for alarm in raised if alarm is not None]

    def _observe_coupling(self, prompt_tokens: int) -> Alarm | None:
        usable = self._usable_window()
        if usable is None:
            return None
        return self._coupling.observe(
            prompt_tokens / usable,
            f"assembled prompt {prompt_tokens} tokens against a usable window "
            f"of {usable} (num_ctx {self._num_ctx} less the num_predict "
            f"{self._num_predict} reply reservation) -- the front of the "
            f"prompt, which carries the <external> untrusted-content rule, is "
            f"what gets evicted first",
        )

    def _usable_window(self) -> int | None:
        """`num_ctx` less the reply reservation, which is never a shed lever.

        Dropping `num_predict` to make a prompt fit measured 4/22 on the
        shipped qwen3:4b, and it fails as an empty reply that logs as success.
        So the reservation is subtracted from the window rather than treated as
        slack, and a prompt is judged against what is actually left for it.

        There is deliberately no fraction applied on top, and the first draft of
        this had one. Carrying the old 0.8 onto the usable window double-counts
        the margin: it breached at 6553 tokens while the boot check certifies a
        worst case near 7500 and passes it, so the one prompt shape this whole
        design is built around would have alarmed on every send. The invariant
        `prompt_tokens + num_predict <= num_ctx` breaches at `usable` exactly,
        and that is already an early warning -- front-truncation does not start
        until `num_ctx`, which is a further `num_predict` tokens away. The
        headroom is the reservation itself; it does not need a second one.
        """
        if self._num_ctx is None:
            return None
        usable = self._num_ctx - (self._num_predict or 0)
        return usable if usable > 0 else None

    def _observe_drift(
        self, prompt_tokens: int, estimated_tokens: int | None
    ) -> Alarm | None:
        if not estimated_tokens or estimated_tokens <= 0:
            return None
        return self._drift.observe(
            prompt_tokens / estimated_tokens,
            f"prompt cost {prompt_tokens} tokens where the boot budget's own "
            f"density predicted at most {estimated_tokens} -- content is "
            f"denser than the worst case that check priced, so the inequality "
            f"it proved does not cover the prompts being sent",
        )
