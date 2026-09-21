"""An id a server has already named is spoken as that name in the confirm
question (DESIGN-voice-confirm.md, "Ids are spoken as names")."""

from __future__ import annotations

import asyncio
from pathlib import Path

from glados.core.adapters import LLMText, LLMToolCall, ToolSpec
from glados.core.config import ClientBinding
from glados.core.product_names import REMEMBERED_PER_SCOPE, ProductNames
from glados.mcp.registry import CallEnvelope, MCPCallResult

from tests.test_voice_confirm import (
    MIC,
    SPEAKER,
    _events,
    _FakeTts,
    _make_org,
    _wait_until_asked,
)

CART = {
    "itemCount": 2,
    "lines": [{"productId": "100806893", "name": "Dunnes Stores Irish Low Fat Milk 3L", "quantity": 2}],
}
ADDED = {"text": "Added 2 x Dunnes Stores Irish Low Fat Milk 3L (productId=100806893) to cart."}
SEARCH_ERROR = (
    "Found 30 matches for 'milk' (top: Dunnes Stores Irish Whole Milk 2L, "
    "productId=100806905) but the add-to-cart button was not clickable."
)


# ---- the memory -------------------------------------------------------------


def test_a_record_with_an_id_and_a_name_is_learned() -> None:
    names = ProductNames()
    assert names.learn("r", "dunnes", CART) == 1
    resolved, named = names.resolve("r", "dunnes", {"productId": "100806893"})
    assert resolved == {"productId": "Dunnes Stores Irish Low Fat Milk 3L"}
    assert [(n.key, n.id, n.name) for n in named] == [
        ("productId", "100806893", "Dunnes Stores Irish Low Fat Milk 3L")
    ]


def test_an_inline_id_after_a_count_or_a_label_is_learned() -> None:
    names = ProductNames()
    assert names.learn("r", "dunnes", ADDED) == 1
    assert names.learn("r", "dunnes", SEARCH_ERROR) == 1
    assert names.resolve("r", "dunnes", {"productId": "100806893"})[0] == {
        "productId": "Dunnes Stores Irish Low Fat Milk 3L"
    }
    assert names.resolve("r", "dunnes", {"productId": 100806905})[0] == {
        "productId": "Dunnes Stores Irish Whole Milk 2L"
    }


def test_an_unknown_id_and_other_args_are_left_alone() -> None:
    names = ProductNames()
    names.learn("r", "dunnes", CART)
    args = {"productId": "1", "quantity": 3, "name": "100806893"}
    assert names.resolve("r", "dunnes", args) == (args, [])


def test_names_do_not_cross_rooms_or_servers() -> None:
    names = ProductNames()
    names.learn("kitchen", "dunnes", CART)
    assert names.resolve("desk", "dunnes", {"productId": "100806893"})[1] == []
    assert names.resolve("kitchen", "tesco", {"productId": "100806893"})[1] == []


def test_nothing_is_learned_from_a_bare_id_or_a_nameless_record() -> None:
    names = ProductNames()
    assert names.learn("r", "s", "productId=100806893") == 0
    assert names.learn("r", "s", {"productId": "100806893", "quantity": 2}) == 0
    assert names.learn("r", "s", {"productId": True, "name": "x"}) == 0
    assert names.learn("r", "s", None) == 0


def test_a_name_is_printable_bounded_and_the_newest_wins() -> None:
    names = ProductNames()
    names.learn("r", "s", {"productId": "7", "name": "old\x07 name"})
    names.learn("r", "s", {"productId": "7", "name": "  new   name " + "x" * 200})
    resolved, _ = names.resolve("r", "s", {"productId": "7"})
    assert resolved["productId"].startswith("new name xxx")
    assert len(resolved["productId"]) == 60


def test_the_memory_is_bounded_per_scope() -> None:
    names = ProductNames()
    for i in range(REMEMBERED_PER_SCOPE + 5):
        names.learn("r", "s", {"productId": str(i), "name": f"item {i}"})
    assert names.resolve("r", "s", {"productId": "0"})[1] == []
    assert names.resolve("r", "s", {"productId": "5"})[1] != []


# ---- the organizer ----------------------------------------------------------


class _LookTool:
    spec = ToolSpec(
        server="t", name="look", description="shows the cart", parameters={"type": "object"}
    )

    async def call(self, args: dict, envelope: CallEnvelope) -> MCPCallResult:
        return MCPCallResult(ok=True, content=CART)


class _LookThenRemoveLLM:
    def __init__(self, look: bool = True) -> None:
        self._n = 0 if look else 1

    async def chat(self, messages, tools):
        self._n += 1
        if self._n == 1:
            yield LLMToolCall(call_id="c0", server="t", name="look", args={})
        elif self._n == 2:
            yield LLMToolCall(
                call_id="c1", server="t", name="boom", args={"productId": "100806893"}
            )
        else:
            yield LLMText(text="done")


async def test_a_seen_id_is_asked_by_name_and_sent_as_the_id(tmp_path: Path) -> None:
    tts = _FakeTts()
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=tts, llm=_LookThenRemoveLLM()) as (
        org,
        _,
        tool,
    ):
        org.mcp.register(_LookTool())
        tool.spec = tool.spec.model_copy(
            update={"confirm_phrase": "remove product {productId} from the cart"}
        )
        await org.handle_user_text("k-mic", "show the cart and remove the milk")
        await _wait_until_asked(org, "kitchen")
        assert tts.spoken == [
            "Say yes to: remove product Dunnes Stores Irish Low Fat Milk 3L "
            "from the cart -- shall I go ahead?"
        ]
        await org.handle_audio_text("k-mic", "Yes.", asyncio.get_running_loop().time())
        await org.flush()
        assert tool.calls == [{"productId": "100806893"}]
        named = next(e for e in _events(tmp_path) if e["event"] == "tool_confirm_id_named")
        assert named["id"] == "100806893" and named["key"] == "productId"
        request = next(e for e in _events(tmp_path) if e["event"] == "tool_confirm_request")
        assert request["tool"] == "t.boom"


async def test_an_id_the_room_never_saw_stays_digits(tmp_path: Path) -> None:
    tts = _FakeTts()
    desk_mic = ClientBinding(client_id="d-mic", room_id="desk", role="mic", default_user="u")
    desk_spk = ClientBinding(
        client_id="d-spk", room_id="desk", role="speaker", default_user="u"
    )
    async with _make_org(
        tmp_path, [MIC, SPEAKER, desk_mic, desk_spk], tts=tts, llm=_LookThenRemoveLLM(look=False)
    ) as (org, _, tool):
        tool.spec = tool.spec.model_copy(
            update={"confirm_phrase": "remove product {productId} from the cart"}
        )
        org._product_names.learn("desk", "t", CART)
        await org.handle_user_text("k-mic", "remove the milk")
        await _wait_until_asked(org, "kitchen")
        assert "1 0 0, 8 0 6, 8 9 3" in tts.spoken[0]
        assert "tool_confirm_id_named" not in [e["event"] for e in _events(tmp_path)]


def test_a_verb_led_report_names_the_id_and_an_unknown_run_up_does_not() -> None:
    names = ProductNames()
    assert names.learn("r", "s", "Removed Dunnes Stores Irish Low Fat Milk 3L (productId=1) from cart.") == 1
    assert names.resolve("r", "s", {"productId": "1"})[0] == {"productId": "Dunnes Stores Irish Low Fat Milk 3L"}
    for text in (
        "Could not add Irish Milk 3L (productId=2): the button was not clickable.",
        "Found Irish Milk 3L (productId=3) in the cart but could not remove it.",
        "timed out waiting for: removing productId=4",
        "The add-back of 2 x productId=5",
        "Removed 6 from cart.",
        "Picked from 30 matches for 'milk'.",
    ):
        assert names.learn("r", "s", text) == 0, text


def test_a_semicolon_joined_list_names_each_id() -> None:
    names = ProductNames()
    text = "Milk 3L (productId=1, qty=2); Eggs 6 pack (productId=555, qty=1)"
    assert names.learn("r", "s", text) == 1
    assert names.resolve("r", "s", {"productId": 555})[0] == {"productId": "Eggs 6 pack"}
    # The first element has no lead in front of it: digits beat a guess.
    assert names.resolve("r", "s", {"productId": 1})[1] == []


def test_a_count_inside_a_name_is_not_a_lead() -> None:
    # Live 19-09-2026: the "6 x" inside the name was taken as the count lead
    # and the question said "remove product 1.5l".
    names = ProductNames()
    names.learn("r", "s", "Added 1 x Volvic Natural Mineral Water 6 x 1.5l (productId=100299422) to cart.")
    assert names.resolve("r", "s", {"productId": "100299422"})[0] == {
        "productId": "Volvic Natural Mineral Water 6 x 1.5l"
    }


def test_a_record_name_is_not_overwritten_by_a_prose_guess() -> None:
    names = ProductNames()
    names.learn("r", "s", {"productId": "9", "name": "Volvic Water 6 x 1.5l"})
    names.learn("r", "s", "Changed the count: 1.5l (productId=9) is now 2.")
    assert names.resolve("r", "s", {"productId": "9"})[0] == {"productId": "Volvic Water 6 x 1.5l"}
    names.learn("r", "s", {"productId": "9", "name": "Volvic Water 6 x 1.5L (new)"})
    assert names.resolve("r", "s", {"productId": "9"})[0] == {"productId": "Volvic Water 6 x 1.5L (new)"}


def test_a_bracket_inside_a_name_is_not_a_lead() -> None:
    names = ProductNames()
    names.learn("r", "s", "Added 2 x Coca-Cola (Diet) 330ml (productId=1) to cart.")
    names.learn("r", "s", "Found 30 matches (top: Ben & Jerry's Cookie Dough (500ml), productId=2) but it failed.")
    assert names.resolve("r", "s", {"productId": "1"})[0] == {"productId": "Coca-Cola (Diet) 330ml"}
    assert names.resolve("r", "s", {"productId": "2"})[0] == {"productId": "Ben & Jerry's Cookie Dough (500ml)"}


async def test_the_dialog_is_told_the_name_beside_the_id(tmp_path: Path) -> None:
    tts = _FakeTts()
    async with _make_org(tmp_path, [MIC, SPEAKER], tts=tts, llm=_LookThenRemoveLLM()) as (
        org,
        sink,
        tool,
    ):
        org.mcp.register(_LookTool())
        await org.handle_user_text("k-mic", "show the cart and remove the milk")
        await _wait_until_asked(org, "kitchen")
        request = next(m for _, m in sink if m["type"] == "tool_confirm_request")
        assert request["args_summary"] == {"productId": "100806893"}
        assert request["arg_names"] == {"productId": "Dunnes Stores Irish Low Fat Milk 3L"}


async def test_the_dialog_gets_no_name_for_an_unseen_id(tmp_path: Path) -> None:
    tts = _FakeTts()
    async with _make_org(
        tmp_path, [MIC, SPEAKER], tts=tts, llm=_LookThenRemoveLLM(look=False)
    ) as (org, sink, tool):
        await org.handle_user_text("k-mic", "remove the milk")
        await _wait_until_asked(org, "kitchen")
        request = next(m for _, m in sink if m["type"] == "tool_confirm_request")
        assert request["arg_names"] == {}


def test_reported_names_reads_the_prose_report_only() -> None:
    from glados.core.product_names import reported_names

    assert reported_names(
        {"text": "Added 1 x Dunnes Stores 6 Organic Apples (productId=100714434) to cart."}
    ) == ("Dunnes Stores 6 Organic Apples",)
    assert reported_names(
        {"lines": [{"productId": "1", "name": "Irish Milk 3L", "quantity": 1}]}
    ) == ()
    assert reported_names(None) == ()


def test_reported_names_ignores_a_cart_echoed_in_prose() -> None:
    from glados.core.product_names import reported_names

    assert reported_names(
        "Added 1 x Apples (productId=1)\nCart now:\nMilk 2L (productId=2)\n"
        "top: Bread, productId=3"
    ) == ("Apples",)
