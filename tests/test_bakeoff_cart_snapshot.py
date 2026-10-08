"""The bake-off runner's cart snapshot: read lines from the view_cart payload
and report what a person must change to restore the starting cart."""

from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load_runner_module():
    sys.path.insert(0, str(_SCRIPTS))
    spec = util.spec_from_file_location("bakeoff_run", _SCRIPTS / "bakeoff_run.py")
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules["bakeoff_run"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()
CartLine = runner.CartLine


def _report(*payloads: tuple[str, object]):
    rep = runner.TurnReport(prompt="view")
    rep.tool_payloads.extend(payloads)
    return rep


_CART = {
    "itemCount": 2,
    "lines": [
        {"productId": "100", "quantity": 2, "name": "Milk 1L"},
        {"productId": "200", "quantity": 6, "packOf": 6, "name": "Eggs x6"},
    ],
}


def test_lines_come_from_the_view_cart_payload():
    lines = runner._cart_lines(_report(("view_cart", _CART)))
    assert lines == [CartLine("100", "Milk 1L", 2), CartLine("200", "Eggs x6", 6)]


def test_lines_are_read_from_a_cart_carried_as_a_json_string():
    lines = runner._cart_lines(_report(("view_cart", {"text": json.dumps(_CART)})))
    assert [line.product_id for line in lines] == ["100", "200"]


def test_the_last_view_cart_wins_and_other_tools_are_ignored():
    rep = _report(
        ("view_cart", _CART),
        ("view_cart", {"lines": []}),
        ("add_to_cart", {"productId": "999", "quantity": 1, "name": "Echo"}),
    )
    assert runner._cart_lines(rep) == []


def test_no_view_cart_call_is_unknown_not_empty():
    assert runner._cart_lines(_report(("add_to_cart", _CART))) is None


@pytest.mark.parametrize(
    "payload",
    [{"error": "session expired"}, "not json", {"text": "You are not logged in."}],
)
def test_a_view_cart_payload_without_lines_is_unknown_not_empty(payload):
    assert runner._cart_lines(_report(("view_cart", payload))) is None


def test_an_empty_lines_list_is_an_empty_cart():
    assert runner._cart_lines(_report(("view_cart", {"itemCount": 0, "lines": []}))) == []


@pytest.mark.parametrize(("raw", "expected"), [("2", 2), ("2.0", 2), (3, 3), (None, 0), ("x", 0)])
def test_quantity_parsing_never_raises(raw, expected):
    assert runner._as_quantity(raw) == expected


def test_an_unchanged_cart_has_no_diff():
    lines = [CartLine("100", "Milk 1L", 2)]
    assert runner._cart_diff(lines, list(lines)) == []


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ([], [CartLine("100", "Milk 1L", 1)], ["Milk 1L (100): 0 at start, 1 now"]),
        ([CartLine("100", "Milk 1L", 2)], [], ["Milk 1L (100): 2 at start, 0 now"]),
        ([CartLine("100", "Milk 1L", 2)], [CartLine("100", "Milk 1L", 3)], ["Milk 1L (100): 2 at start, 3 now"]),
    ],
)
def test_each_changed_product_is_reported(before, after, expected):
    assert runner._cart_diff(before, after) == expected


def test_duplicate_lines_of_one_product_are_summed():
    before = [CartLine("100", "Milk 1L", 2)]
    after = [CartLine("100", "Milk 1L", 1), CartLine("100", "Milk 1L", 1)]
    assert runner._cart_diff(before, after) == []


def test_a_dirty_cart_refuses_and_lists_its_lines():
    with pytest.raises(SystemExit, match=r"Milk 1L \(100\) x2"):
        runner._refuse_dirty_cart([CartLine("100", "Milk 1L", 2)])
