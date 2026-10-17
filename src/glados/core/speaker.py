"""Speaker identity contract (DESIGN-speaker-id.md).

`SpeakerIdentity` is the one type other components key on. `person` is an
enrolled person or None -- never a guess. `session_key` is only the bucket a
session is filed under: for an unidentified voice it falls back to the
client's `default_user`, which is a real person, so nothing may use it to
choose whose data to touch. Memory reads and writes require `verified`.

Not an authorisation input: no permission, confirm or user-scoped write may
branch on any of this until v6 (ARCHITECTURE.md section 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .wake import Admission

SpeakerSource = Literal["voice", "fallback", "typed"]

MatchReason = Literal[
    "accepted",
    "low_score",
    "low_margin",
    "too_short",
    "timeout",
    "busy",
    "error",
    "no_prints",
    "off",
]


@dataclass(frozen=True)
class SpeakerMatch:
    """What voice identification concluded about one utterance. Carries no
    embedding, so it is safe to trace."""

    person: str | None
    score: float | None
    reason: MatchReason

    @classmethod
    def none(cls, reason: MatchReason, score: float | None = None) -> "SpeakerMatch":
        return cls(person=None, score=score, reason=reason)


NO_IDENTIFIER = SpeakerMatch.none("off")


@dataclass(frozen=True)
class SpeakerIdentity:
    person: str | None
    session_key: str
    source: SpeakerSource
    score: float | None = None

    @property
    def verified(self) -> bool:
        return self.source == "voice"


@dataclass(frozen=True)
class Utterance:
    """One transcribed utterance with the identity heard in the SAME audio.
    Ingress tracks add fields here rather than callback arguments."""

    text: str
    captured_at: float
    speaker: SpeakerMatch = NO_IDENTIFIER
    # Set when wake gating admitted the segment (DESIGN.md, wake word).
    admission: Admission | None = None


def resolve_identity(match: SpeakerMatch | None, default_user: str) -> SpeakerIdentity:
    """`match` None means the turn was typed; a SpeakerMatch means it was
    spoken, identified or not."""
    if match is None:
        return SpeakerIdentity(person=default_user, session_key=default_user, source="typed")
    if match.person is None:
        return SpeakerIdentity(person=None, session_key=default_user, source="fallback")
    return SpeakerIdentity(
        person=match.person, session_key=match.person, source="voice", score=match.score
    )
