"""Human scoring of the shadow review file (DESIGN-memory-scorer.md, Part B).

The server appends facts to `review.jsonl`; this module never rewrites it.
Scores go to an append-only sidecar holding only the line id and the score,
so no fact text spreads past the file `forget` and `wipe` already know about.
The latest score for an id wins, which makes a re-score an append.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

from ..core.memory_review import _restrict_to_owner

SCORES = ("2", "1", "0", "x")
SCORE_MEANING = {"2": "useful", "1": "harmless", "0": "noise", "x": "wrong"}
UNSCORED = "-"


@dataclass(frozen=True)
class ReviewItem:
    id: str
    run_date: str
    room: str
    person: str
    kind: str
    fact: str
    untrusted: bool


def sidecar_path(review_path: Path) -> Path:
    return review_path.with_name("review.scores.jsonl")


def load_items(path: Path) -> list[ReviewItem]:
    return [_item(line, row) for line, row in _rows(path)]


def load_scores(path: Path) -> dict[str, str]:
    scores: dict[str, str] = {}
    for _, row in _rows(path):
        if row.get("score") in SCORES and isinstance(row.get("id"), str):
            scores[row["id"]] = row["score"]
    return scores


def append_score(path: Path, item_id: str, score: str) -> None:
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": item_id, "score": score, "at": stamp}) + "\n")
    _restrict_to_owner(path)


def pending(items: Iterable[ReviewItem], scores: dict[str, str], person: str) -> list[ReviewItem]:
    return [i for i in items if i.person == person and i.id not in scores]


def summarise(items: Iterable[ReviewItem], scores: dict[str, str]) -> dict[tuple[str, str, str], Counter]:
    """Score counts keyed by (week, person, kind); unscored counts as `-`."""
    table: dict[tuple[str, str, str], Counter] = {}
    for item in items:
        key = (_week(item.run_date), item.person, item.kind)
        table.setdefault(key, Counter())[scores.get(item.id, UNSCORED)] += 1
    return table


def format_summary(table: dict[tuple[str, str, str], Counter]) -> str:
    columns = (*SCORES, UNSCORED)
    header = "week\tperson\tkind\t" + "\t".join(columns) + "\tuseful%"
    lines = [header]
    for key in sorted(table):
        counts = table[key]
        cells = "\t".join(str(counts[c]) for c in columns)
        lines.append("\t".join(key) + f"\t{cells}\t{_useful_share(counts)}")
    return "\n".join(lines)


def run_review(
    items: list[ReviewItem], sidecar: Path, ask: Callable[[str], str], say: Callable[[str], None],
) -> int:
    """Ask for a score per item until the list ends or the person quits.
    Returns how many were scored."""
    scored = 0
    for n, item in enumerate(items, start=1):
        answer = _ask_score(item, f"[{n}/{len(items)}]", ask, say)
        if answer == "q":
            break
        if answer in SCORES:
            append_score(sidecar, item.id, answer)
            scored += 1
    return scored


def _ask_score(item: ReviewItem, progress: str, ask: Callable[[str], str], say: Callable[[str], None]) -> str:
    say(f"{progress} {item.run_date} {item.kind}{' (untrusted)' if item.untrusted else ''}: {_printable(item.fact)}")
    while True:
        answer = ask("2 useful / 1 harmless / 0 noise / x wrong / s skip / q quit > ").strip().lower()
        if answer in (*SCORES, "s", "q"):
            return answer
        say("type 2, 1, 0, x, s or q")


def _rows(path: Path) -> Iterable[tuple[str, dict]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = _parse(line)
            if row is not None:
                yield line, row


def _parse(line: str) -> dict | None:
    if not line.endswith("\n"):
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


def _item(line: str, row: dict) -> ReviewItem:
    return ReviewItem(
        id=row.get("id") or _legacy_id(line),
        run_date=str(row.get("run_date", "")),
        room=str(row.get("room", "")),
        person=str(row.get("person", "")),
        kind=str(row.get("kind", "")),
        fact=str(row.get("fact", "")),
        untrusted=bool(row.get("untrusted", False)),
    )


def _legacy_id(line: str) -> str:
    return "h" + hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()[:31]


def _week(run_date: str) -> str:
    try:
        year, week, _ = date.fromisoformat(run_date).isocalendar()
    except ValueError:
        return "unknown"
    return f"{year}-W{week:02d}"


def _useful_share(counts: Counter) -> str:
    scored = sum(counts[s] for s in SCORES)
    return f"{100 * counts['2'] // scored}" if scored else UNSCORED


def _printable(text: str) -> str:
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)
