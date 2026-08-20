"""Unit tests for the vendor-neutral spoken-length cap.

Two properties matter most and are tested hardest: the withheld count must be
derived from the list actually sliced, so the number GLaDOS speaks is one it
can stand over; and NOTHING here may raise -- a malformed scrape has to degrade
to today's long-list behaviour, never to a dead turn or a silent "nothing
found", which reads exactly like the truth."""

from __future__ import annotations

import pytest

from glados.core.tool_payload_cap import PayloadCap, cap_tool_payload


def _items(n: int) -> list[dict]:
    return [{"name": f"item-{i}"} for i in range(n)]


# ---- bare-list payloads -------------------------------------------------


def test_short_list_is_untouched() -> None:
    payload = _items(3)
    capped = cap_tool_payload(payload, PayloadCap(max_items=5))
    assert capped.content == payload
    assert capped.withheld == 0
    assert not capped.capped


def test_list_at_the_cap_is_untouched() -> None:
    payload = _items(5)
    assert cap_tool_payload(payload, PayloadCap(max_items=5)).content == payload


def test_long_list_is_truncated_and_counted() -> None:
    capped = cap_tool_payload(_items(12), PayloadCap(max_items=5))
    assert capped.content["shown"] == _items(12)[:5]
    assert capped.content["withheld_count"] == 7
    assert capped.withheld == 7
    assert capped.capped


def test_order_is_preserved_exactly() -> None:
    """Ranking belongs to the server that owns the data -- the cap slices, it
    never re-sorts."""
    capped = cap_tool_payload(_items(9), PayloadCap(max_items=5))
    assert [i["name"] for i in capped.content["shown"]] == [
        "item-0", "item-1", "item-2", "item-3", "item-4",
    ]


# ---- flex ---------------------------------------------------------------


@pytest.mark.parametrize("total", [6, 7])
def test_flex_speaks_the_whole_list_without_a_remainder(total: int) -> None:
    """Offering "want the other one?" is absurd out loud -- up to flex_to the
    list is spoken whole and no offer is made."""
    payload = _items(total)
    capped = cap_tool_payload(payload, PayloadCap(max_items=5, flex_to=7))
    assert capped.content == payload
    assert capped.withheld == 0


def test_one_past_flex_truncates_to_the_hard_cap() -> None:
    capped = cap_tool_payload(_items(8), PayloadCap(max_items=5, flex_to=7))
    assert len(capped.content["shown"]) == 5
    assert capped.content["withheld_count"] == 3


def test_flex_below_the_cap_is_incoherent_and_disables_capping() -> None:
    payload = _items(20)
    assert cap_tool_payload(payload, PayloadCap(max_items=5, flex_to=3)).content == payload


# ---- keyed payloads -----------------------------------------------------


def test_nested_list_is_capped_and_siblings_survive() -> None:
    payload = {"items": _items(12), "fakeSaleCount": 4}
    capped = cap_tool_payload(payload, PayloadCap(max_items=5, items_key="items"))
    assert len(capped.content["items"]) == 5
    assert capped.content["withheld_count"] == 7
    assert capped.content["fakeSaleCount"] == 4


def test_the_input_payload_is_not_mutated() -> None:
    payload = {"items": _items(12)}
    cap_tool_payload(payload, PayloadCap(max_items=5, items_key="items"))
    assert len(payload["items"]) == 12


def test_withheld_count_is_derived_from_the_list_it_sliced() -> None:
    """Never from a total the server reported -- a count computed off the raw
    scrape would offer items the server had already filtered out."""
    payload = {"items": _items(8), "totalScanned": 14}
    capped = cap_tool_payload(payload, PayloadCap(max_items=5, items_key="items"))
    assert capped.content["withheld_count"] == 3


# ---- degradation: none of these may raise -------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        "a string",
        42,
        {"items": None},
        {"items": "not a list"},
        {"wrong_key": _items(12)},
        [],
    ],
)
def test_unrecognised_shapes_pass_through_untouched(payload) -> None:
    capped = cap_tool_payload(payload, PayloadCap(max_items=5, items_key="items"))
    assert capped.content == payload
    assert capped.withheld == 0


def test_a_bare_list_is_ignored_when_a_key_was_declared() -> None:
    payload = _items(12)
    assert cap_tool_payload(payload, PayloadCap(max_items=5, items_key="items")).content == payload


def test_a_dict_is_ignored_when_no_key_was_declared() -> None:
    payload = {"items": _items(12)}
    assert cap_tool_payload(payload, PayloadCap(max_items=5)).content == payload


def test_items_need_not_be_dicts() -> None:
    capped = cap_tool_payload(["a", "b", "c"], PayloadCap(max_items=2))
    assert capped.content["shown"] == ["a", "b"]
    assert capped.withheld == 1


@pytest.mark.parametrize("max_items", [0, -1])
def test_a_nonsense_cap_disables_capping(max_items: int) -> None:
    payload = _items(12)
    assert cap_tool_payload(payload, PayloadCap(max_items=max_items)).content == payload


# ---- schema drift is distinguishable from a short list ------------------


def test_a_short_list_still_counts_as_recognised() -> None:
    """A cap that did nothing because nothing was needed must not look like a
    cap that could not find the list -- only the latter is worth a warning."""
    assert cap_tool_payload({"items": _items(2)}, PayloadCap(max_items=5, items_key="items")).recognised


def test_a_missing_key_is_not_recognised() -> None:
    assert not cap_tool_payload({"nope": _items(9)}, PayloadCap(max_items=5, items_key="items")).recognised
