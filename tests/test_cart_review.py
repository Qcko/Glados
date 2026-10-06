"""The display-only checkout review parse (core/cart_review.py)."""

from __future__ import annotations

import pytest

from glados.core.cart_review import CartReview

MILK = "100806893"
COMBINING_ACUTE = chr(0x0301)


def _cart(name: str = "Milk", **extra) -> dict:
    return {"lines": [{"productId": MILK, "name": name, "quantity": 2}], **extra}


def test_reads_lines_and_totals_as_strings() -> None:
    review = CartReview.from_content(_cart(orderValue=12.5, itemCount=2))
    assert review.payload.line_count == 1
    assert review.payload.order_value == "12.50"
    assert review.payload.item_count == 2


def test_a_long_name_is_clipped_by_code_points() -> None:
    review = CartReview.from_content(_cart(name="x" * 10_000))
    assert len(review.payload.lines[0].name) <= 120


def test_a_tower_of_combining_marks_is_clipped_by_graphemes() -> None:
    name = ("a" + COMBINING_ACUTE) * 100
    shown = CartReview.from_content(_cart(name=name)).payload.lines[0].name
    assert sum(1 for ch in shown if ch == "a") <= 60


@pytest.mark.parametrize("bad", [-1, 10_000, 1.234, "12.50", True, float("inf")])
def test_an_off_contract_total_rejects_the_whole_review(bad) -> None:
    assert CartReview.from_content(_cart(orderValue=bad)) is None


def test_a_name_that_is_not_text_rejects_the_whole_review() -> None:
    assert CartReview.from_content(_cart(name=5)) is None


def test_the_digest_binds_quantities_and_totals_but_not_names() -> None:
    base = CartReview.from_content(_cart(orderValue=6))
    renamed = CartReview.from_content(_cart(name="Other", orderValue=6))
    repriced = CartReview.from_content(_cart(orderValue=7))
    assert base.digest == renamed.digest
    assert base.digest != repriced.digest
