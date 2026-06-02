"""Deterministic routing rules: text in, a local/cloud verdict out.

Per ARCHITECTURE.md v2.6 the rules are also the **privacy gate** — §9's
invariant tightens to "no cloud LLM call carries data the user hasn't
explicitly opted into routing externally", and these rules are what decide a
turn crosses that boundary. So they err toward *local*: only requests that
clearly want open-ended reasoning (which the local 14b fumbles) are sent out.

The verdict carries a `confidence`. A `low`-confidence local verdict means the
rules couldn't classify cleanly; the organizer's escalation path can retry such
a turn on cloud if its deterministic outcome comes back `failed`. A `high`
verdict is the rules committing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ...core.turn_outcome import is_action_request

RouteTarget = Literal["local", "cloud"]
RouteConfidence = Literal["high", "low"]

# Markers of open-ended reasoning the local model is known to fumble:
# explanation, comparison, justification, recommendation. Matched as whole
# words anywhere in the request — unlike action verbs these aren't
# position-sensitive ("milk or oat, which is better?" should still route out).
_CLOUD_MARKERS = re.compile(
    r"\b(?:why|explain|compare|comparison|difference|differences|versus|vs|"
    r"pros and cons|trade-?offs?|analy[sz]e|elaborate|reason|justify|"
    r"recommend|suggest|advise|opinion|think about|how come|"
    r"what if|should i|worth it)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    target: RouteTarget
    reason: str
    confidence: RouteConfidence


@dataclass(frozen=True)
class Router:
    """Deterministic per-turn router. `max_words_local` is the length above
    which a request is treated as multi-clause reasoning and sent to cloud."""

    max_words_local: int = 30

    def decide(self, text: str) -> RouteDecision:
        stripped = (text or "").strip()
        if not stripped:
            return RouteDecision("local", "empty request", "high")
        if _CLOUD_MARKERS.search(stripped):
            return RouteDecision("cloud", "reasoning/comparison markers", "high")
        words = stripped.split()
        if len(words) > self.max_words_local:
            return RouteDecision(
                "cloud", f"long request ({len(words)} words)", "high"
            )
        if is_action_request(stripped):
            return RouteDecision("local", "tool-trigger imperative", "high")
        if len(words) <= 6:
            return RouteDecision("local", "short utterance", "high")
        # Mid-length, no clear markers — the local brain is the cheap, private
        # default, but the rules aren't sure. Flagged low so a `failed` outcome
        # can escalate rather than the user just getting a bad local answer.
        return RouteDecision("local", "ambiguous — default local", "low")
