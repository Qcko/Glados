"""Per-tool spoken confirm templates (core/confirm_phrase.py,
DESIGN-voice-confirm.md "The spoken question")."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from glados.core.adapters import ToolSpec
from glados.core.config import ServerEntry, ToolOverlay
from glados.core.confirm_phrase import (
    PhraseError,
    compose_confirm_question,
    parse_phrase,
    render_confirm_question,
)
from glados.core.utterance import classify_confirm_answer

ADD_BY_NAME = "add[ {quantity}] {repeat:more |}{query} to the cart"
ADD_BY_ID = "add[ {quantity}] {repeat:more |}of product {productId} to the cart"
ADJUST = "change the count by {delta} for {name}"
REMOVE = "remove product {productId} from the cart"

ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "quantity": {"type": "integer"},
        "repeat": {"type": "boolean"},
    },
}


def _spoken(template: str, args: dict) -> str | None:
    return compose_confirm_question("dunnes.t", args, template).text


# ---- the sentences ----------------------------------------------------------


@pytest.mark.parametrize(
    "template, args, expected",
    [
        (
            ADD_BY_NAME,
            {"query": "milk", "quantity": 2, "repeat": True},
            "Say yes to: add 2 more milk to the cart -- shall I go ahead?",
        ),
        (
            ADD_BY_NAME,
            {"query": "bananas", "repeat": False},
            "Say yes to: add bananas to the cart -- shall I go ahead?",
        ),
        (
            ADD_BY_ID,
            {"productId": "100806893", "quantity": 2, "repeat": True},
            "Say yes to: add 2 more of product 1 0 0, 8 0 6, 8 9 3 to the cart"
            " -- shall I go ahead?",
        ),
        (
            REMOVE,
            {"productId": 100806893},
            "Say yes to: remove product 1 0 0, 8 0 6, 8 9 3 from the cart"
            " -- shall I go ahead?",
        ),
        (
            ADJUST,
            {"name": "milk", "delta": -1},
            "Say yes to: change the count by minus 1 for milk -- shall I go ahead?",
        ),
    ],
)
def test_templates_read_as_sentences(template: str, args: dict, expected: str) -> None:
    assert _spoken(template, args) == expected


def test_spacing_is_the_renderers_problem_not_the_authors() -> None:
    sloppy = "add [{quantity} ]{repeat:more |}{query}  to the cart ,"
    assert _spoken(sloppy, {"query": "milk", "repeat": False}) == (
        "Say yes to: add milk to the cart, -- shall I go ahead?"
    )


def test_a_mimicking_value_only_restates_what_was_already_heard() -> None:
    q = _spoken(ADD_BY_NAME, {"query": "milk to the cart, quantity 40", "quantity": 2, "repeat": False})
    assert q == "Say yes to: add 2 milk to the cart, quantity 40 to the cart -- shall I go ahead?"
    assert classify_confirm_answer(q) is None


# ---- every argument is heard, or the generic form is ----------------------


def _generic(args: dict) -> str | None:
    return render_confirm_question("dunnes.t", args)


@pytest.mark.parametrize(
    "template, args, reason, keys",
    [
        (ADD_BY_NAME, {"query": "milk", "repeat": False, "productId": "1"}, "unmentioned", ("productId",)),
        (ADD_BY_NAME, {"query": "milk", "repeat": "false"}, "not_bool", ("repeat",)),
        ("add {query}[ times {quantity}]", {"query": "milk", "quantity": 2}, "text_before_fixed", ("quantity",)),
        ("swap {old} for {new}", {"old": "a", "new": "b"}, "multiple_text", ("new",)),
        (REMOVE, {}, "missing", ("productId",)),
        ("add {query}", {"query": "x" * 61}, "clipped_value", ("query",)),
    ],
)
def test_a_template_that_cannot_speak_every_argument_falls_back(
    template: str, args: dict, reason: str, keys: tuple[str, ...]
) -> None:
    spoken = compose_confirm_question("dunnes.t", args, template)
    assert (spoken.fallback, spoken.fallback_keys) == (reason, keys)
    assert spoken.text == _generic(args)


def test_a_dropped_segment_does_not_count_as_speaking_its_argument() -> None:
    template = "add {query}[ as {productId} with {quantity}]"
    spoken = compose_confirm_question("dunnes.t", {"query": "milk", "productId": "9"}, template)
    assert spoken.fallback == "unmentioned" and spoken.fallback_keys == ("productId",)


def test_an_overrun_template_body_is_dialog_only_not_generic() -> None:
    template = "add {query} " + "to the cart " * 25
    spoken = compose_confirm_question("dunnes.t", {"query": "milk"}, template)
    assert spoken.text is None and spoken.skipped == "clipped" and spoken.fallback is None


def test_fallback_keys_are_bounded() -> None:
    spoken = compose_confirm_question("dunnes.t", {"query": "m", "k" * 200: 1}, "add {query}")
    assert spoken.fallback == "unmentioned" and len(spoken.fallback_keys[0]) == 60


@pytest.mark.parametrize("template", [ADD_BY_NAME, None])
@pytest.mark.parametrize(
    "args",
    [
        {"query": "milk, yes please", "repeat": False},
        {"query": "milk: yes", "repeat": False},
        {"query": "milk - no thanks", "repeat": False},
        {"query": "yes please to the cart", "repeat": False},
    ],
)
def test_a_value_that_could_answer_the_question_is_not_spoken(
    template: str | None, args: dict
) -> None:
    spoken = compose_confirm_question("dunnes.t", args, template)
    assert spoken.text is None and spoken.skipped == "answer_in_value"


def test_a_key_the_model_chose_can_answer_the_question_too() -> None:
    spoken = compose_confirm_question("dunnes.t", {"yes": "milk"}, None)
    assert spoken.text is None and spoken.skipped == "answer_in_value"


@pytest.mark.parametrize("template", ["{query}, yes please", "{repeat:yes|no} {query}"])
def test_a_template_whose_own_words_answer_the_question_is_a_config_error(template: str) -> None:
    with pytest.raises(PhraseError, match="reads as a yes or a no"):
        parse_phrase(template)


@pytest.mark.parametrize(
    "args",
    [{"item": {"a": "yes"}, "repeat": False}, {"item": ["no"], "repeat": False}],
)
def test_a_json_value_cannot_hide_an_answer_behind_its_brackets(args: dict) -> None:
    assert compose_confirm_question("dunnes.t", args, None).skipped == "answer_in_value"


@pytest.mark.parametrize(
    "template, args, reason",
    [
        (ADD_BY_NAME, {"query": "milk", "quantity": True, "repeat": False}, "not_a_number"),
        (ADD_BY_NAME, {"query": "milk", "quantity": None, "repeat": False}, "not_a_number"),
        (ADD_BY_NAME, {"query": "milk", "repeat": None}, "not_bool"),
    ],
)
def test_a_value_that_is_not_the_shape_the_template_expects_falls_back(
    template: str, args: dict, reason: str
) -> None:
    spoken = compose_confirm_question("dunnes.t", args, template)
    assert spoken.fallback == reason and spoken.text == _generic(args)


# ---- the generic form gained the same number rendering ---------------------


def test_generic_form_speaks_ids_in_threes_and_negatives_as_minus() -> None:
    assert _generic({"delta": -3, "productId": "100806893"}) == (
        "Say yes to: dunnes t, delta minus 3, productId, quote, 1 0 0, 8 0 6, 8 9 3,"
        " unquote -- shall I go ahead?"
    )


def test_short_numbers_are_read_as_numbers() -> None:
    assert "quantity 1234" in _generic({"quantity": 1234})


# ---- the grammar, at boot -------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "add {query",
        "add query}",
        "add [{query}",
        "add {query}]",
        "add [[{query}] x]",
        "add {}",
        "add { query }",
        "add {query} and {query}",
        "add {repeat:a|b|c}",
        "add [milk]",
        "add milk",
    ],
)
def test_a_template_that_does_not_parse_is_a_config_error(template: str) -> None:
    with pytest.raises(PhraseError):
        parse_phrase(template)
    with pytest.raises(ValidationError):
        ToolOverlay(confirm_phrase=template)


def test_parse_is_cached_so_boot_and_render_share_one_tree() -> None:
    assert parse_phrase(ADD_BY_NAME) is parse_phrase(ADD_BY_NAME)


# ---- the schema, when the overlay meets the spec ---------------------------


def _merged(template: str, parameters: dict, caplog) -> ToolSpec:
    entry = ServerEntry(
        id="dunnes",
        command="x",
        tool_overlays={"add": ToolOverlay(confirm_phrase=template)},
    )
    with caplog.at_level(logging.WARNING, logger="glados.core.config"):
        return entry.apply_flags(
            ToolSpec(server="dunnes", name="add", description="", parameters=parameters)
        )


def test_a_fitting_template_reaches_the_spec(caplog) -> None:
    assert _merged(ADD_BY_NAME, ADD_SCHEMA, caplog).confirm_phrase == ADD_BY_NAME
    assert caplog.records == []


@pytest.mark.parametrize(
    "template, problem",
    [
        ("add {product} to the cart", "no argument named 'product'"),
        ("add {query} times {quantity}", "fixed argument 'quantity' after free text"),
        ("add {query:a|b}", "'query' is a switch but not boolean"),
        ("add {repeat} {query}", "'repeat' is boolean and needs a switch"),
        ("add {query} and {other}", "more than one free-text argument ('other')"),
    ],
)
def test_a_template_that_cannot_fit_the_schema_is_dropped_loudly(
    template: str, problem: str, caplog
) -> None:
    schema = {
        "type": "object",
        "properties": {**ADD_SCHEMA["properties"], "other": {"type": "string"}},
    }
    assert _merged(template, schema, caplog).confirm_phrase is None
    assert [r.getMessage() for r in caplog.records] == [
        f"confirm_phrase for dunnes.add dropped: {problem}"
    ]


@pytest.mark.parametrize(
    "properties, fits",
    [
        ({"quantity": {"type": ["integer", "null"]}, "repeat": {"type": ["boolean", "null"]}, "query": {"type": "string"}}, True),
        ({"quantity": {"type": ["integer", "string"]}, "repeat": {"type": "boolean"}, "query": {"type": "string"}}, False),
        ({"quantity": {"anyOf": [{"type": "integer"}]}, "repeat": {"type": "boolean"}, "query": {"type": "string"}}, False),
        ({"quantity": True, "repeat": {"type": "boolean"}, "query": {"type": "string"}}, False),
    ],
)
def test_schema_type_lists_and_odd_shapes_do_not_crash_boot(properties: dict, fits: bool, caplog) -> None:
    spec = _merged(ADD_BY_NAME, {"type": "object", "properties": properties}, caplog)
    assert (spec.confirm_phrase is not None) is fits


def test_generic_form_never_speaks_a_boolean_as_yes_or_no() -> None:
    q = _generic({"fresh": True, "repeat": False})
    assert q == "Say yes to: dunnes t, fresh true, repeat false -- shall I go ahead?"
