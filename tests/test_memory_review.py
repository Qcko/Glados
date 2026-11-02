"""Shadow review scoring CLI (DESIGN-memory-scorer.md, Part B)."""

from __future__ import annotations

import json

import pytest

from glados.memory import review
from glados.memory.__main__ import main


def _line(**over) -> str:
    row = {"at": "2026-10-20T03:00:00", "id": "a1", "run_date": "2026-10-20", "room": "desk",
           "person": "qcko", "session": "s1", "kind": "add", "fact": "Is vegetarian.",
           "untrusted": False, "score": None}
    row.update(over)
    return json.dumps(row) + "\n"


@pytest.fixture
def review_file(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text(
        _line()
        + _line(id="a2", fact="Has a cat.", kind="reaffirm")
        + _line(id="b1", person="other", fact="Plays chess.")
        + _line(id="a3", run_date="2026-10-27", kind="drop", fact="Lives in Cork."),
        encoding="utf-8",
    )
    return path


def test_a_trailing_partial_line_is_skipped(review_file):
    with review_file.open("a", encoding="utf-8") as f:
        f.write(_line(id="half").rstrip("\n")[:40])
    assert [i.id for i in review.load_items(review_file)] == ["a1", "a2", "b1", "a3"]


def test_a_legacy_line_without_id_gets_a_stable_id(tmp_path):
    path = tmp_path / "review.jsonl"
    row = json.loads(_line())
    del row["id"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    first = review.load_items(path)[0].id
    assert first.startswith("h") and review.load_items(path)[0].id == first


def test_pending_shows_only_that_persons_unscored_facts(review_file):
    sidecar = review.sidecar_path(review_file)
    review.append_score(sidecar, "a1", "2")
    todo = review.pending(review.load_items(review_file), review.load_scores(sidecar), "qcko")
    assert [i.id for i in todo] == ["a2", "a3"]


def test_the_latest_score_wins_and_the_sidecar_holds_no_fact_text(review_file):
    sidecar = review.sidecar_path(review_file)
    review.append_score(sidecar, "a1", "0")
    review.append_score(sidecar, "a1", "2")
    assert review.load_scores(sidecar) == {"a1": "2"}
    assert "vegetarian" not in sidecar.read_text(encoding="utf-8")


def test_a_bad_score_is_refused(tmp_path):
    with pytest.raises(ValueError):
        review.append_score(tmp_path / "s.jsonl", "a1", "3")


def test_run_review_scores_skips_reprompts_and_quits(review_file):
    sidecar = review.sidecar_path(review_file)
    items = review.pending(review.load_items(review_file), {}, "qcko")
    answers = iter(["7", "x", "s", "q"])
    said: list[str] = []
    scored = review.run_review(items, sidecar, lambda _: next(answers), said.append)
    assert scored == 1 and review.load_scores(sidecar) == {"a1": "x"}
    assert any("type 2, 1, 0, x, s or q" in s for s in said)


def test_control_characters_in_a_fact_are_shown_escaped(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text(_line(fact="Likes\x1b[2Jtea."), encoding="utf-8")
    said: list[str] = []
    review.run_review(review.load_items(path), tmp_path / "s.jsonl", lambda _: "q", said.append)
    assert "\x1b" not in said[0] and "\\x1b" in said[0]


def test_summary_counts_per_week_person_and_kind(review_file):
    scores = {"a1": "2", "a2": "0"}
    table = review.summarise(review.load_items(review_file), scores)
    assert table[("2026-W43", "qcko", "add")] == {"2": 1}
    assert table[("2026-W43", "qcko", "reaffirm")] == {"0": 1}
    assert table[("2026-W44", "qcko", "drop")] == {"-": 1}
    text = review.format_summary(table)
    assert text.splitlines()[0].startswith("week\tperson\tkind")
    assert "2026-W43\tqcko\tadd\t1\t0\t0\t0\t0\t100" in text


def test_cli_review_and_summary(review_file, monkeypatch, capsys):
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["review", "--person", "qcko", "--file", str(review_file)]) == 0
    assert "scored 1 of 3" in capsys.readouterr().out
    assert main(["summary", "--file", str(review_file)]) == 0
    assert "2026-W43\tqcko\tadd\t0\t1\t0\t0\t0\t0" in capsys.readouterr().out


def test_cli_review_with_nothing_pending(review_file, capsys):
    assert main(["review", "--person", "nobody", "--file", str(review_file)]) == 0
    assert "nothing to score for nobody" in capsys.readouterr().out


def test_drop_and_reaffirm_lines_carry_untrusted_from_their_notes():
    from glados.core.memory_facts import Fact, WriterVerdict
    from glados.core.memory_writer import review_entries
    from glados.core.speaker_memory import ChunkWrite

    write = ChunkWrite(group=("desk", "qcko", "s1"), run_date="2026-10-20", note_ids=(1, 2), expected=(0, 0))
    verdict = WriterVerdict(adds=[], drop_ids=[10], rejected=0, reaffirmed={11: (1,), 12: (2,)})
    known = [Fact(10, "Lives in Cork."), Fact(11, "Has a cat."), Fact(12, "Plays chess.")]
    rows = {r["fact"]: r for r in review_entries(write, verdict, known, frozenset({2}))}
    assert rows["Lives in Cork."]["untrusted"] is True
    assert rows["Has a cat."]["untrusted"] is False
    assert rows["Plays chess."]["untrusted"] is True


def test_a_drop_in_a_fully_trusted_chunk_is_trusted():
    from glados.core.memory_facts import Fact, WriterVerdict
    from glados.core.memory_writer import review_entries
    from glados.core.speaker_memory import ChunkWrite

    write = ChunkWrite(group=("desk", "qcko", "s1"), run_date="2026-10-20", note_ids=(1,), expected=(0, 0))
    verdict = WriterVerdict(adds=[], drop_ids=[10], rejected=0)
    [row] = review_entries(write, verdict, [Fact(10, "Lives in Cork.")])
    assert row["untrusted"] is False
