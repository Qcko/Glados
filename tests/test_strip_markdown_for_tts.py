"""Unit tests for the strip-for-TTS pass.

Two jobs are under test: markdown tokens must never reach Piper as literal
characters ("asterisk asterisk milk asterisk asterisk"), and a line that WAS a
bullet or a heading must come out with sentence punctuation so the synthesized
pause lands between items instead of mid-phrase. Real punctuation the user
would hear is not ours to touch."""

from __future__ import annotations

import pytest

from glados.core.organizer import _strip_markdown_for_tts


# ---- inline emphasis ----------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("**milk** is on sale", "milk is on sale"),
        ("*milk* is on sale", "milk is on sale"),
        ("***milk*** is on sale", "milk is on sale"),
        ("__milk__ is on sale", "milk is on sale"),
        ("_milk_ is on sale", "milk is on sale"),
        ("`milk` is on sale", "milk is on sale"),
        ("``milk`` is on sale", "milk is on sale"),
    ],
)
def test_inline_markers_are_removed(text: str, expected: str) -> None:
    assert _strip_markdown_for_tts(text) == expected


def test_intra_word_underscores_survive() -> None:
    assert _strip_markdown_for_tts("scan_favorites_for_sales ran") == (
        "scan_favorites_for_sales ran"
    )


def test_plain_prose_is_untouched() -> None:
    text = "I added four things. Two were out of stock; one was not."
    assert _strip_markdown_for_tts(text) == text


def test_prose_without_a_terminator_is_not_given_one() -> None:
    assert _strip_markdown_for_tts("milk is on sale") == "milk is on sale"


# ---- bullets ------------------------------------------------------------


@pytest.mark.parametrize("marker", ["-", "*", "+"])
def test_bullet_markers_become_a_full_stop(marker: str) -> None:
    assert _strip_markdown_for_tts(f"{marker} milk") == "milk."


@pytest.mark.parametrize("marker", ["1.", "2)", "10."])
def test_ordered_list_markers_are_left_alone(marker: str) -> None:
    """An enumeration is content, not formatting -- the listener wants to hear
    "one", and the number already gives Piper its stop."""
    assert _strip_markdown_for_tts(f"{marker} milk") == f"{marker} milk"


def test_indented_bullet_is_stripped() -> None:
    assert _strip_markdown_for_tts("    - milk") == "milk."


def test_bullet_list_gets_a_terminator_per_item() -> None:
    text = "- milk\n- cheddar\n- sourdough"
    assert _strip_markdown_for_tts(text) == "milk.\ncheddar.\nsourdough."


@pytest.mark.parametrize("ending", [".", "!", "?", ":"])
def test_existing_full_stop_is_not_doubled(ending: str) -> None:
    assert _strip_markdown_for_tts(f"- milk{ending}") == f"milk{ending}"


@pytest.mark.parametrize("glue", [",", ";"])
def test_trailing_list_glue_is_promoted_to_a_full_stop(glue: str) -> None:
    """A comma is the short run-on pause this change exists to remove."""
    assert _strip_markdown_for_tts(f"- milk{glue}") == "milk."


def test_trailing_whitespace_before_the_terminator_is_trimmed() -> None:
    assert _strip_markdown_for_tts("- milk   ") == "milk."


def test_a_bare_bullet_marker_yields_an_empty_line() -> None:
    assert _strip_markdown_for_tts("- \nmilk") == "\nmilk"


# ---- headings -----------------------------------------------------------


@pytest.mark.parametrize("hashes", ["#", "##", "######"])
def test_heading_markers_become_a_full_stop(hashes: str) -> None:
    assert _strip_markdown_for_tts(f"{hashes} On sale") == "On sale."


def test_hash_without_a_space_is_not_a_heading() -> None:
    assert _strip_markdown_for_tts("#1 seller") == "#1 seller"


# ---- combinations -------------------------------------------------------


def test_emphasis_inside_a_bullet_is_stripped_before_the_terminator() -> None:
    assert _strip_markdown_for_tts("- **milk** 2 for 3 euro") == (
        "milk 2 for 3 euro."
    )


def test_a_mixed_reply_reads_as_sentences() -> None:
    text = "## On sale\n- **milk**\n- cheddar\nThat is all."
    assert _strip_markdown_for_tts(text) == (
        "On sale.\nmilk.\ncheddar.\nThat is all."
    )


def test_blank_lines_are_preserved() -> None:
    assert _strip_markdown_for_tts("- milk\n\n- cheddar") == (
        "milk.\n\ncheddar."
    )


def test_empty_input_stays_empty() -> None:
    assert _strip_markdown_for_tts("") == ""
