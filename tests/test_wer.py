# rule-guard:allow ascii-source - curly quotes are the fixture text the normaliser is asserted to fold.
"""Unit tests for the standalone WER implementation.

Cases chosen to exercise each error class independently (so a regression
in `_edit_counts` or `_backtrace` shows up as a wrong category, not just
a wrong total)."""

from __future__ import annotations

import pytest

from glados.audio.stt.wer import WerResult, tokenize, wer


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert tokenize("It's fine.") == ["it's", "fine"]
    assert tokenize("   leading   spaces   ") == ["leading", "spaces"]
    assert tokenize("") == []
    assert tokenize("!!!") == []


def test_perfect_match_is_zero_wer() -> None:
    r = wer("the quick brown fox", "the quick brown fox")
    assert r == WerResult(wer=0.0, substitutions=0, deletions=0, insertions=0,
                          ref_words=4, hyp_words=4)


def test_substitution_only() -> None:
    r = wer("the quick brown fox", "the slow brown fox")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 0)
    assert r.wer == pytest.approx(0.25)


def test_deletion_only() -> None:
    r = wer("the quick brown fox", "the brown fox")
    assert (r.substitutions, r.deletions, r.insertions) == (0, 1, 0)
    assert r.wer == pytest.approx(0.25)


def test_insertion_only() -> None:
    r = wer("the brown fox", "the quick brown fox")
    assert (r.substitutions, r.deletions, r.insertions) == (0, 0, 1)
    assert r.wer == pytest.approx(1 / 3)


def test_mixed_errors() -> None:
    # ref: a b c d e
    # hyp: a x c e      -> sub b->x, del d
    r = wer("a b c d e", "a x c e")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 1, 0)
    assert r.wer == pytest.approx(2 / 5)


def test_empty_reference_with_hyp_is_full_error() -> None:
    r = wer("", "stray words")
    assert r.wer == 1.0
    assert r.insertions == 2


def test_empty_reference_and_hyp_is_zero() -> None:
    r = wer("", "")
    assert r.wer == 0.0


def test_punctuation_does_not_count_as_error() -> None:
    r = wer("hello world", "Hello, world!")
    assert r.wer == 0.0


def test_curly_apostrophe_normalised_to_straight() -> None:
    # Common Voice ships curly U+2019; faster-whisper emits ASCII.
    # They must tokenise to the same word, not a substitution.
    assert tokenize("don’t stop") == tokenize("don't stop")
    r = wer("don’t stop", "don't stop")
    assert r.wer == 0.0
