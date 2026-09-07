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

**Who measures and who judges.** `PromptEstimator` lives on the adapter and
prices a prompt; `PromptPressureMonitor` judges, and the organizer owns one per
(model, session) via `SessionPressureMonitors`. That split is not decoration.
The first version put both on the adapter, and one adapter serves every room --
so a streak of "three consecutive breaches" could be assembled from three
unrelated conversations, and a clean send in any room re-armed an alarm another
room was building toward. The word "consecutive" claimed more than the code
delivered. An adapter cannot fix this itself: `chat()` takes messages and tools
and has no idea what a conversation is.

The numbers reach the organizer as an `LLMUsage` event riding the stream, NOT
as a field on the adapter. That distinction is load-bearing rather than
stylistic: rooms run concurrently on one shared adapter, so any per-adapter slot
is read by whichever turn arrives first, and the second attempt at this fix
traded a cross-session streak bug for a cross-session attribution bug by
forgetting it. An event is a per-call local by construction.

Nothing here decides policy or changes a prompt. It reports; the caller logs,
traces, or ignores. A monitor that could shed would be a second shedding path
disagreeing with `Organizer._shed_for_hop`, and the ceiling that was measured
is the one that should bind.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .adapters import LLMMessage, LLMUsage
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

    Not thread-safe, and does not need to be: `SessionPressureMonitors` hands
    each (model, session) its own instance, and a session's turns are serialised
    by its room's queue worker. "Consecutive" therefore means consecutive sends
    WITHIN one conversation on one brain, which is what the word should have
    meant all along.
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


class PromptEstimator:
    """The adapter's half: price a prompt before it is sent.

    Holds the boot budget and nothing else. It deliberately does NOT judge --
    an adapter is shared by every room it serves, so any streak it kept would
    be a streak across unrelated conversations. Judging lives with the thing
    that knows what a conversation is.
    """

    def __init__(self) -> None:
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


class PromptPressureMonitor:
    """The judging half, owned per CONVERSATION rather than per adapter.

    This is the fix for a defect the first version shipped with and documented
    instead of correcting. The monitor used to live on the adapter, and one
    adapter serves every room, so "three consecutive breaches" could be
    assembled from three unrelated conversations -- and one clean send in any
    room re-armed an alarm another room was building toward. The word
    "consecutive" was doing work the code did not do.

    Everything it needs now arrives in the `LLMUsage`, so the organizer
    can keep one of these per session without knowing anything about a
    backend's configuration. `num_ctx` may be None (llama.cpp's real window is
    `llama-server`'s launch `-c`, which the adapter cannot see): the coupling
    check then goes quiet rather than guessing, while drift, which needs no
    window at all, keeps working.
    """

    def __init__(self, *, streak: int = DEFAULT_STREAK) -> None:
        self._coupling = StreakAlarm("context_pressure", streak=streak)
        self._drift = StreakAlarm("estimator_drift", streak=streak)

    def observe(self, reading: LLMUsage | None) -> list[Alarm]:
        if reading is None:
            return []
        raised = [
            self._observe_coupling(reading),
            self._observe_drift(reading),
        ]
        return [alarm for alarm in raised if alarm is not None]

    def _observe_coupling(self, reading: LLMUsage) -> Alarm | None:
        usable = _usable_window(reading)
        if usable is None:
            return None
        return self._coupling.observe(
            reading.prompt_tokens / usable,
            f"assembled prompt {reading.prompt_tokens} tokens against a usable "
            f"window of {usable} (num_ctx {reading.num_ctx} less the "
            f"num_predict {reading.num_predict} reply reservation) -- the "
            f"front of the prompt, which carries the <external> "
            f"untrusted-content rule, is what gets evicted first",
        )

    def _observe_drift(self, reading: LLMUsage) -> Alarm | None:
        estimated = reading.estimated_tokens
        if not estimated or estimated <= 0:
            return None
        return self._drift.observe(
            reading.prompt_tokens / estimated,
            f"prompt cost {reading.prompt_tokens} tokens where the boot "
            f"budget's own density predicted at most {estimated} -- content is "
            f"denser than the worst case that check priced, so the inequality "
            f"it proved does not cover the prompts being sent",
        )


def _usable_window(reading: LLMUsage) -> int | None:
    """`num_ctx` less the reply reservation, which is never a shed lever.

    Dropping `num_predict` to make a prompt fit measured 4/22 on the shipped
    qwen3:4b, and it fails as an empty reply that logs as success. So the
    reservation is subtracted from the window rather than treated as slack, and
    a prompt is judged against what is actually left for it.

    There is deliberately no fraction applied on top, and the first draft of
    this had one. Carrying the old 0.8 onto the usable window double-counts the
    margin: it breached at 6553 tokens while the boot check certifies a worst
    case near 7500 and passes it, so the one prompt shape this whole design is
    built around would have alarmed on every send. The invariant
    `prompt_tokens + num_predict <= num_ctx` breaches at `usable` exactly, and
    that is already an early warning -- front-truncation does not start until
    `num_ctx`, which is a further `num_predict` tokens away. The headroom is the
    reservation itself; it does not need a second one.
    """
    if reading.num_ctx is None:
        return None
    usable = reading.num_ctx - (reading.num_predict or 0)
    return usable if usable > 0 else None


class SessionPressureMonitors:
    """One `PromptPressureMonitor` per conversation, with a bound on how many.

    The bound is not paranoia about leaks so much as about a long-lived process:
    sessions are created freely and never announce that they are finished, so an
    unbounded dict here would be a slow one. Eviction is least-recently-used and
    costs only a forgotten streak, which is the cheapest thing here to lose -- a
    re-breach simply starts counting again.
    """

    def __init__(self, *, streak: int = DEFAULT_STREAK, max_sessions: int = 64) -> None:
        self._streak = streak
        self._max_sessions = max(1, max_sessions)
        self._monitors: dict[tuple[str, str], PromptPressureMonitor] = {}

    def observe(self, session_id: str, reading: LLMUsage | None) -> list[Alarm]:
        if reading is None:
            return []
        if not _can_judge(reading):
            # Neither check can fire on this reading, so a monitor for it would
            # be an entry that never decides anything -- and on a backend that
            # reports no window and has no boot budget, that is every entry.
            return []
        monitor = self._monitor_for((reading.model, session_id))
        return monitor.observe(reading)

    def _monitor_for(self, key: tuple[str, str]) -> PromptPressureMonitor:
        """Keyed by MODEL and session, not session alone.

        A session that escalates runs `_pick_brain`'s specialist adapter under
        the same `session_id` as the primary -- different model, different
        `num_ctx`, different reply reservation. Keying on the session alone
        would blend the two into one streak and let a clean specialist send
        re-arm a breach the primary was building toward, which is the very
        defect this class exists to fix, along the brain axis instead of the
        room axis.
        """
        monitor = self._monitors.pop(key, None)
        if monitor is None:
            monitor = PromptPressureMonitor(streak=self._streak)
        # Re-inserted on every access, so the dict is ordered least-recently-
        # used and eviction takes the coldest key. Insertion order alone would
        # evict the room that has been running since boot -- the busiest one,
        # and so the one most likely to be mid-breach -- the moment a burst of
        # short-lived sessions arrived.
        self._monitors[key] = monitor
        while len(self._monitors) > self._max_sessions:
            self._monitors.pop(next(iter(self._monitors)))
        return monitor


def _can_judge(reading: LLMUsage) -> bool:
    """Whether either check could say anything at all about this reading."""
    return _usable_window(reading) is not None or bool(reading.estimated_tokens)
