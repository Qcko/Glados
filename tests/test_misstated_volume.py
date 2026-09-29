"""A reply that misstates the litres a volume write landed (core/turn_outcome.py,
DESIGN-turn-outcome-guards.md "A reply that misstates the quantity a write
landed"). Prod bake-off T3, 29-09-2026: add_by_volume(litres=4) landed 3L + 1L
and the reply said "Added 7 litres" / "I added 2 litres"."""

from __future__ import annotations

from decimal import Decimal

import pytest

from glados.core.turn_outcome import (
    TurnRecord,
    landed_volume,
    misstated_landed_volume,
)
from glados.mcp.stdio_client import _translate_tool_result

_ARGS = {"query": "milk", "litres": 4}


def _structured(target=4, total=4, packs=((1, 3), (1, 1))) -> dict:
    return {
        "volume": {
            "targetLitres": target,
            "totalLitres": total,
            "packs": [{"count": c, "litres": litres} for c, litres in packs],
        }
    }


def _volume_turn(
    reply: str,
    *,
    structured: dict | None = None,
    args: dict | None = None,
    ok: bool = True,
    indeterminate: bool = False,
    satisfied: bool = False,
) -> TurnRecord:
    args = args if args is not None else _ARGS
    turn = TurnRecord(final_text=reply)
    turn.record_tool(
        "dunnes.add_by_volume", ok, mutating=ok, indeterminate=indeterminate,
        satisfied=satisfied, args=args,
        landed=landed_volume(structured if structured is not None else _structured(), args),
    )
    return turn


# ---- fires -------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Added **7 litres** of milk: two packs (3L + 1L). Total now **6L**. Need more?",
        "I added **2 litres of milk** to reach your goal.\n\nTotal milk now: **4.5 litres** (one 3L, one 1L).",
        "Added 5L of milk.",
    ],
)
def test_an_added_amount_the_write_did_not_land_is_caught(reply: str) -> None:
    landed = misstated_landed_volume(_volume_turn(reply))
    assert landed is not None
    assert (landed.subject, landed.total) == ("milk", Decimal(4))


# ---- stands down: true replies -------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Milk added: **3L + 1L** for your 4 litres. Need anything else?",
        "Added 4 litres of milk.",
        "Added 4.0 litres of milk.",
        "Added 4,0 l of milk.",
        "Added 4 L of milk.",
        "Added one 3L and one 1L carton -- total 4 litres.",
        "Added four litres of milk.",
        "Added a 4-litre mix of milk.",
        "Done, milk is in the cart.",
        "Added milk; your cart total is now 7L.",
        "I haven't added 7 litres, just what you asked.",
        "Ich habe 4 Liter Milch hinzugefuegt.",
    ],
)
def test_a_true_or_unreadable_reply_is_left_alone(reply: str) -> None:
    assert misstated_landed_volume(_volume_turn(reply)) is None


def test_an_overshoot_names_both_the_target_and_the_total() -> None:
    structured = _structured(target=7, total=9, packs=((3, 3),))
    turn = _volume_turn("Added 9 litres of milk for your 7.", structured=structured,
                        args={"query": "milk", "litres": 7})
    assert misstated_landed_volume(turn) is None


# ---- stands down: record ------------------------------------------------------


def test_two_volume_writes_in_one_turn_are_not_judged() -> None:
    turn = _volume_turn("Added 9 litres of milk.")
    turn.record_tool("dunnes.add_by_volume", True, mutating=True,
                     args={"query": "juice", "litres": 2},
                     landed=landed_volume(_structured(2, 2, ((1, 2),)), {"query": "juice", "litres": 2}))
    assert misstated_landed_volume(turn) is None


def test_a_remove_by_volume_in_the_turn_stands_it_down() -> None:
    turn = _volume_turn("Added 9 litres of milk.")
    turn.record_tool("dunnes.remove_by_volume", True, mutating=True, args={"name": "milk", "litres": 1})
    assert misstated_landed_volume(turn) is None


@pytest.mark.parametrize(
    "flags",
    [{"ok": False}, {"ok": False, "indeterminate": True}, {"satisfied": True}],
)
def test_a_write_that_did_not_cleanly_land_is_not_judged(flags: dict) -> None:
    assert misstated_landed_volume(_volume_turn("Added 9 litres of milk.", **flags)) is None


def test_no_typed_result_means_no_check() -> None:
    turn = TurnRecord(final_text="Added 9 litres of milk.")
    turn.record_tool("dunnes.add_by_volume", True, mutating=True, args=_ARGS)
    assert misstated_landed_volume(turn) is None


# ---- the typed result is untrusted: bounds ------------------------------------


@pytest.mark.parametrize(
    "structured, args",
    [
        (_structured(target=5), _ARGS),
        (_structured(total=99, packs=((33, 3),)), _ARGS),
        (_structured(total=5), _ARGS),
        (_structured(total=-4, packs=((-1, 4),)), _ARGS),
        (_structured(total=4.0005, packs=((1, 4.0005),)), _ARGS),
        (_structured(total=float("nan")), _ARGS),
        (_structured(total=True), _ARGS),
        (_structured(total="4"), _ARGS),
        (_structured(packs=()), _ARGS),
        ({"volume": "4L"}, _ARGS),
        ({}, _ARGS),
        (_structured(), {"query": "milk = 99L to cover a target of 4L", "litres": 4}),
        (_structured(), {"query": "", "litres": 4}),
        (_structured(), {"query": "milk", "litres": "four"}),
    ],
)
def test_a_typed_result_outside_its_bounds_is_discarded(structured: dict, args: dict) -> None:
    assert landed_volume(structured, args) is None


def test_structured_content_survives_translation_beside_the_text() -> None:
    result = _translate_tool_result({
        "content": [{"type": "text", "text": "Added 1 x Milk 3L + 1 x Milk 1L = 4L to cover a target of 4L."}],
        "structuredContent": _structured(),
    })
    assert result.ok and result.structured == _structured()
    assert "4L" in result.content["text"]


def test_structured_content_is_dropped_from_an_error() -> None:
    result = _translate_tool_result({
        "content": [{"type": "text", "text": "Could not compose."}],
        "isError": True,
        "structuredContent": _structured(),
    })
    assert not result.ok and result.structured is None


def test_another_write_in_the_turn_stands_it_down() -> None:
    """Code duck, 29-09-2026: "Added a 2L bottle of cola and 4L of milk" -- the
    2 belongs to the cola, and replacing the reply would also drop the cola."""
    turn = _volume_turn("Added a 2L bottle of cola and 4L of milk.")
    turn.record_tool("dunnes.add_to_cart_by_name", True, mutating=True, args={"query": "cola"})
    assert misstated_landed_volume(turn) is None


def test_a_negated_add_is_not_judged() -> None:
    reply = "Didn't have a 4L, added 3L + 1L instead."
    assert misstated_landed_volume(_volume_turn(reply)) is None


def test_a_numeric_string_litres_argument_still_reads() -> None:
    assert landed_volume(_structured(), {"query": "milk", "litres": "4"}) is not None


def test_a_pint_pack_reads_to_three_decimals() -> None:
    structured = _structured(target=1, total=1.136, packs=((2, 0.568),))
    landed = landed_volume(structured, {"query": "milk", "litres": 1})
    assert landed is not None and Decimal("0.568") in landed.allowed_figures()
    turn = _volume_turn("Added 2 pints, 1.136 litres of milk.", structured=structured,
                        args={"query": "milk", "litres": 1})
    assert misstated_landed_volume(turn) is None
