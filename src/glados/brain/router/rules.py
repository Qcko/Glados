"""Deterministic routing rules: text in, a primary/specialist verdict out.

Per ARCHITECTURE.md §12 (v2.6) the router picks which **local** brain handles a
turn — the primary model, or a resident specialist kept for the open-ended
reasoning the primary fumbles. Both run on the home server, so routing crosses
no privacy boundary; the rules simply err toward the cheap *primary* path and
only send a turn to the specialist when it clearly wants open-ended reasoning.
(The specialist slot can be backed by the dormant cloud escape hatch, but that
is a provider choice in the wiring, not something these rules decide.)

The verdict carries a `confidence`. A `low`-confidence primary verdict means the
rules couldn't classify cleanly; the organizer's escalation path can retry such
a turn on the specialist if its deterministic outcome comes back `failed`. A
`high` verdict is the rules committing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ...core.turn_outcome import is_action_request

RouteTarget = Literal["primary", "specialist"]
RouteConfidence = Literal["high", "low"]

# Markers of open-ended reasoning the primary model is known to fumble:
# explanation, comparison, justification, recommendation. Matched as whole
# words anywhere in the request — unlike action verbs these aren't
# position-sensitive ("milk or oat, which is better?" should still route out).
_SPECIALIST_MARKERS = re.compile(
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
    which a request is treated as multi-clause reasoning and sent to the
    specialist."""

    max_words_local: int = 30

    def decide(self, text: str) -> RouteDecision:
        stripped = (text or "").strip()
        if not stripped:
            return RouteDecision("primary", "empty request", "high")
        if _SPECIALIST_MARKERS.search(stripped):
            return RouteDecision(
                "specialist", "reasoning/comparison markers", "high"
            )
        # An imperative to mutate state is a tool-trigger the primary handles
        # well — check it BEFORE the length gate. Otherwise a long multi-item
        # add ("add: <18 ingredients>") routes to the specialist purely on word
        # count, where the small model truncates the list and over-claims
        # "done" — the history-poisoning seed (SESSION 2026-06-16).
        if is_action_request(stripped):
            return RouteDecision("primary", "tool-trigger imperative", "high")
        words = stripped.split()
        if len(words) > self.max_words_local:
            return RouteDecision(
                "specialist", f"long request ({len(words)} words)", "high"
            )
        if len(words) <= 6:
            return RouteDecision("primary", "short utterance", "high")
        # Mid-length, no clear markers — the primary brain is the cheap default,
        # but the rules aren't sure. Flagged low so a `failed` outcome can
        # escalate to the specialist rather than the user just getting a bad
        # primary answer.
        return RouteDecision("primary", "ambiguous — default primary", "low")
