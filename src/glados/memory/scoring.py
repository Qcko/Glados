"""Pure scoring for the memory-writer bake-off (DESIGN-memory-scorer.md).

No I/O and no model: a case says what a day of notes should leave behind, an
outcome says what the real fold did leave behind, and this module compares
the two. Matchers are keyword sets over normalised words, so a verdict is
deterministic and a human can read why it was given.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

SPLITS = ("dev", "test")
_NOT_WORD = re.compile(r"[^a-z0-9]+")


class CaseError(ValueError):
    pass


@dataclass(frozen=True)
class Matcher:
    """`all`: every phrase present; `any`: at least one; `none`: no phrase
    present. `none` is what stops "is not vegetarian" or "partner is
    vegetarian" from counting as "vegetarian"."""

    all: tuple[str, ...] = ()
    any: tuple[str, ...] = ()
    none: tuple[str, ...] = ()

    def matches(self, fact: str) -> bool:
        text = f" {_words(fact)} "
        present = [f" {_words(p)} " in text for p in (*self.all, *self.any, *self.none)]
        n_all, n_any = len(self.all), len(self.any)
        return (
            all(present[:n_all])
            and (not n_any or any(present[n_all:n_all + n_any]))
            and not any(present[n_all + n_any:])
        )


@dataclass(frozen=True)
class Note:
    said: str
    reply: str | None = None
    untrusted: bool = False


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    origin: str
    tags: tuple[str, ...]
    known: tuple[str, ...]
    notes: tuple[Note, ...]
    expect_add: tuple[Matcher, ...] = ()
    expect_drop: tuple[str, ...] = ()
    forbid: tuple[Matcher, ...] = ()
    also_ok: tuple[Matcher, ...] = ()
    max_notes_per_batch: int | None = None

    @property
    def empty_day(self) -> bool:
        """Nothing should change: the most common real day."""
        return not self.expect_add and not self.expect_drop


@dataclass(frozen=True)
class Outcome:
    """What the fold left behind for one case."""

    added: tuple[str, ...]
    dropped: tuple[str, ...]
    elapsed_ms: int = 0
    # Writer calls that errored (HTTP, OOM, timeout). A failed call adds
    # nothing, which would otherwise score as a perfect empty day.
    failed_chunks: int = 0


@dataclass
class CaseScore:
    case_id: str
    split: str
    empty_day: bool
    expected: int
    matched: int
    drops_expected: int
    drops_correct: int
    drops_wrong: int
    forbidden: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    failed_chunks: int = 0


def parse_case(raw: dict) -> Case:
    try:
        case = Case(
            id=str(raw["id"]),
            split=raw["split"],
            origin=raw["origin"],
            tags=tuple(raw.get("tags", ())),
            known=tuple(raw.get("known", ())),
            notes=tuple(Note(**n) for n in raw["notes"]),
            expect_add=_matchers(raw.get("expect_add", ())),
            expect_drop=tuple(raw.get("expect_drop", ())),
            forbid=_matchers(raw.get("forbid", ())),
            also_ok=_matchers(raw.get("also_ok", ())),
            max_notes_per_batch=raw.get("max_notes_per_batch"),
        )
    except (KeyError, TypeError) as e:
        raise CaseError(f"case {raw.get('id')!r}: {e}") from e
    _check(case)
    return case


def _matchers(raw) -> tuple[Matcher, ...]:
    return tuple(Matcher(**{k: tuple(v) for k, v in m.items()}) for m in raw)


def _check(case: Case) -> None:
    if case.split not in SPLITS:
        raise CaseError(f"case {case.id!r}: split must be one of {SPLITS}")
    if not case.notes:
        raise CaseError(f"case {case.id!r}: needs at least one note")
    missing = [d for d in case.expect_drop if d not in case.known]
    if missing:
        raise CaseError(f"case {case.id!r}: expect_drop not in known: {missing}")


def load_cases(text: str) -> list[Case]:
    cases = [parse_case(json.loads(line)) for line in text.splitlines() if line.strip()]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CaseError(f"duplicate case ids: {duplicates}")
    return cases


def score_case(case: Case, outcome: Outcome) -> CaseScore:
    added = sorted(outcome.added)
    matched_ids = _match_one_to_one(case.expect_add, added)
    leftover = [f for i, f in enumerate(added) if i not in matched_ids]
    forbidden = [f for f in added if any(m.matches(f) for m in case.forbid)]
    unexpected = [
        f for f in leftover
        if f not in forbidden and not any(m.matches(f) for m in case.also_ok)
    ]
    dropped = set(outcome.dropped)
    return CaseScore(
        case_id=case.id,
        split=case.split,
        empty_day=case.empty_day,
        expected=len(case.expect_add),
        matched=len(matched_ids),
        drops_expected=len(case.expect_drop),
        drops_correct=len(dropped & set(case.expect_drop)),
        drops_wrong=len(dropped - set(case.expect_drop)),
        forbidden=forbidden,
        unexpected=unexpected,
        elapsed_ms=outcome.elapsed_ms,
        failed_chunks=outcome.failed_chunks,
    )


def _match_one_to_one(expectations: tuple[Matcher, ...], facts: list[str]) -> set[int]:
    """Each fact satisfies at most one expectation. Greedy over sorted facts,
    which is order-stable; bipartite matching is on the deferred list."""
    used: set[int] = set()
    for matcher in expectations:
        hit = next((i for i, f in enumerate(facts) if i not in used and matcher.matches(f)), None)
        if hit is not None:
            used.add(hit)
    return used


@dataclass(frozen=True)
class Summary:
    cases: int
    forbidden_cases: tuple[str, ...]
    failed_cases: tuple[str, ...]
    recall: float
    precision: float
    f1: float
    empty_days: int
    empty_day_false_adds: float
    unadjudicated: int
    p90_ms: int

    @property
    def gate(self) -> str:
        """"pass", "VETO" or "INVALID". A forbidden hit is a veto, never
        averaged away; a failed writer call or an empty split makes the run
        invalid, never a pass."""
        if self.failed_cases or not self.cases:
            return "INVALID"
        return "VETO" if self.forbidden_cases else "pass"

    @property
    def gate_passed(self) -> bool:
        return self.gate == "pass"


def summarise(scores: list[CaseScore]) -> Summary:
    """Unadjudicated facts are neither right nor wrong: they are counted on
    their own and kept out of precision, so a model is not penalised for a
    valid paraphrase nobody has judged yet."""
    correct = sum(s.matched + s.drops_correct for s in scores)
    wanted = sum(s.expected + s.drops_expected for s in scores)
    wrong = sum(len(s.forbidden) + s.drops_wrong for s in scores)
    recall = correct / wanted if wanted else 1.0
    precision = correct / (correct + wrong) if correct + wrong else 1.0
    empty = [s for s in scores if s.empty_day]
    noisy = [s for s in empty if s.forbidden or s.unexpected or s.drops_wrong]
    return Summary(
        cases=len(scores),
        forbidden_cases=tuple(s.case_id for s in scores if s.forbidden),
        failed_cases=tuple(s.case_id for s in scores if s.failed_chunks),
        recall=recall,
        precision=precision,
        f1=_f1(precision, recall),
        empty_days=len(empty),
        empty_day_false_adds=len(noisy) / len(empty) if empty else 0.0,
        unadjudicated=sum(len(s.unexpected) for s in scores),
        p90_ms=_p90([s.elapsed_ms for s in scores]),
    )


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _p90(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]


def _words(text: str) -> str:
    return _NOT_WORD.sub(" ", text.lower()).strip()
