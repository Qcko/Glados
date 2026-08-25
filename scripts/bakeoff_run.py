"""Model bake-off runner -- drives MODEL_BAKE_OFF.md's prompts through a
running GLaDOS server and prints a structured per-test report.

It does NOT score quality -- that stays a human judgement (the whole point of
the bake-off). What it automates is the tedious, error-prone part: sending each
test prompt in the documented order, and capturing what came back --
tool calls + args, tool results (ok/error), the spoken reply, and the
deterministic `turn_outcome` (done / needs-user / failed). You read the report
and fill in the 0/1/2 scorecard.

The model under test is whatever the running server is configured for -- swap
it the documented way (un/comment `[llm] model` in configs/glados.toml, then
`glados-stop` + `uv run glados`) and re-run this with the matching `--slot`
label. The `--slot` arg only tags the output; it does not change the model.

Usage:
    uv run python scripts/bakeoff_run.py --slot B
    uv run python scripts/bakeoff_run.py --slot A --url wss://127.0.0.1:8765/ws/v1

Auth: the token is read from the OS keyring (scope `glados.client-tokens`,
username = client id), the same store the server uses. Override with --token
for a dev fixture.

Caveat -- turn-to-turn memory: T4 ("actually add eggs instead") and T8 ("now
add it back") depend on conversational memory the v0 organizer does not yet
have (single-turn sessions). They are sent anyway so you can observe the
behaviour, but flagged in the report as memory-dependent.

Stateful-cart pollution & --reset: the cart tests (T6/T8/T10/T11/T12) share one
server-side cart with no reset between them, so an earlier test pollutes a later
one -- e.g. T3's add_by_volume can leave both a 3 L and a 1 L milk line, after
which every later "milk" op hits the NameResolver non-unique guard and can't
score a clean pass. Pass --reset to empty the cart and re-establish a known
single-milk starting state before the stateful block (anchored to the test
flagged `reset_before`). The reset is driven through the same ws/v1 user_text
path the suite already uses (no direct-tool frame exists), so it is itself
LLM-mediated; it is logged under a distinct CART RESET banner so a run that
reset state is visibly distinguishable from one that did not. Without --reset
the suite behaves exactly as before. Note the Dunnes cart is persisted per
logged-in account and survives GLaDOS restarts, so it must be cleared
explicitly -- restarting the server does not reset it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field

import websockets

from _ws import ssl_context
from glados.core.secrets import KeyringSecrets


@dataclass
class BakeoffTest:
    id: str
    prompts: list[str]
    criterion: str
    memory_dependent: bool = False
    # Part of the shared-cart stateful block. When --reset is passed, the cart
    # is emptied and reseeded to a known single-milk state once, before the
    # first stateful test that actually runs (so it works under --only too).
    stateful: bool = False


# Prompts that drive the cart back to a known starting state through the same
# ws/v1 user_text path the suite uses. There is no client-side direct-tool
# frame, so the reset is itself LLM-mediated -- phrased to remove *every* line
# (by product id, unambiguous) rather than a per-name op that would trip the
# non-unique guard, then seed a single milk line the stateful tests expect.
RESET_PROMPTS: list[str] = [
    "Call view_cart to list the lines, then for each line call remove_from_cart "
    "using that line's real productId value copied verbatim from the view_cart "
    "result -- never a placeholder. Repeat until view_cart shows an empty cart.",
    "Add one carton of milk to the cart.",
]


# Order follows MODEL_BAKE_OFF.md "Run order note": T9 first (authenticates the
# browser session T2-T8 ride on), then T1 (independent), then T2-T8.
TESTS: list[BakeoffTest] = [
    BakeoffTest(
        "T9",
        ["Use the dunnes start_browser tool with headless equals true, then check if I'm logged in."],
        "executes both calls in sequence (two different tools, one turn, no asking).",
    ),
    BakeoffTest("T1", ["What time is it?"], "calls time.now, speaks a time, one pass."),
    BakeoffTest(
        "T2",
        ["Add milk to the cart."],
        "picks add_to_cart_by_name(query='milk', quantity=1), single tool call, real productId.",
    ),
    BakeoffTest(
        "T3",
        ["Add 4 liters of milk to the cart."],
        "ends with 4 L total volume in cart (1x3L+1x1L or 4x1L).",
    ),
    BakeoffTest(
        "T4",
        ["Add bananas to the cart.", "Actually, add eggs instead."],
        "ends the turn with eggs in cart, not both.",
        memory_dependent=True,
    ),
    BakeoffTest(
        "T5",
        ["Which of my favorites are on sale?"],
        "lists only promo items; describes IsRealSale=null as unverified.",
    ),
    BakeoffTest(
        "T6",
        ["Show me what's in my cart and then remove the milk."],
        "view_cart then remove/ set_cart_quantity on the milk line; >=2 calls, one turn.",
        stateful=True,
    ),
    BakeoffTest(
        "T8",
        ["Now add it back."],
        "re-adds the milk by the productId it just removed, without re-searching.",
        memory_dependent=True,
        stateful=True,
    ),
    BakeoffTest(
        "T10",
        ["Add two more milks to the cart."],
        "treats 'more' as ADDITIVE: add_to_cart / add_to_cart_by_name with "
        "quantity 2 (cart milk count goes UP by 2), not a set-to-2. Assumes "
        "milk is already in the cart (run T2 first).",
        memory_dependent=True,
        stateful=True,
    ),
    BakeoffTest(
        "T11",
        ["Actually, just make it one milk."],
        "treats 'make it N' as an ABSOLUTE set: resolves the milk productId "
        "(view_cart / search) then set_cart_quantity(id, 1) -- NOT add_to_cart. "
        "Ends with milk quantity 1.",
        memory_dependent=True,
        stateful=True,
    ),
    BakeoffTest(
        "T12",
        ["Take one of the milks off."],
        "RELATIVE reduce: view_cart to read the current milk quantity, then "
        "set_cart_quantity(id, current-1) (or remove if it was 1). One user "
        "turn, >=2 tool calls. Hardest -- read-then-compute-then-act.",
        memory_dependent=True,
        stateful=True,
    ),
]


@dataclass
class TurnReport:
    prompt: str
    deltas: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    route: str | None = None
    outcome: str | None = None
    ended: str = "done"

    @property
    def reply(self) -> str:
        return "".join(self.deltas).strip()


async def _run_turn(ws, prompt: str) -> TurnReport:
    """Send one user_text and collect every server frame until the turn ends.
    Auto-grants tool confirmations so gated tools proceed unattended."""
    rep = TurnReport(prompt=prompt)
    await ws.send(json.dumps({"type": "user_text", "text": prompt}))
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
        except asyncio.TimeoutError:
            rep.ended = "timeout (no terminal frame in 120s)"
            return rep
        except websockets.ConnectionClosed:
            rep.ended = "connection closed"
            return rep
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "assistant_delta":
            rep.deltas.append(msg["text"])
        elif kind == "tool_call":
            args = json.dumps(msg.get("args", {}), ensure_ascii=False)
            rep.tool_calls.append(f"{msg['server']}.{msg['name']}({args})")
        elif kind == "tool_result":
            rep.tool_results.append(
                "ok" if msg["ok"] else f"ERROR: {msg.get('error')}"
            )
        elif kind == "route_notice":
            tag = "escalated->cloud" if msg["escalated"] else msg["target"]
            rep.route = f"{tag} ({msg['reason']})"
        elif kind == "turn_outcome":
            rep.outcome = msg["outcome"]
        elif kind == "tool_confirm_request":
            await ws.send(json.dumps({
                "type": "tool_confirm_response",
                "request_id": msg["request_id"],
                "granted": True,
            }))
        elif kind in ("done", "cancelled", "error"):
            rep.ended = kind if kind != "error" else f"error: {msg.get('message')}"
            return rep


async def _reset_cart(ws) -> None:
    """Drive the cart back to a known single-milk state via RESET_PROMPTS.
    LLM-mediated (same user_text path as the suite); logged under a distinct
    banner so a reset run is visibly distinguishable from one that isn't."""
    print("=== CART RESET (--reset) -- empty + seed single milk ===")
    for prompt in RESET_PROMPTS:
        rep = await _run_turn(ws, prompt)
        _print_turn(rep)
    print("=== CART RESET done ===")


def _print_turn(rep: TurnReport) -> None:
    print(f"    prompt : {rep.prompt}")
    if rep.route:
        print(f"    route  : {rep.route}")
    for tc in rep.tool_calls:
        print(f"    tool-> : {tc}")
    for tr in rep.tool_results:
        print(f"    <-tool : {tr}")
    print(f"    reply  : {rep.reply or '(none)'}")
    flag = "" if rep.ended == "done" else f"  [ended: {rep.ended}]"
    print(f"    OUTCOME: {rep.outcome or '(none)'}{flag}")


async def run(args: argparse.Namespace) -> None:
    token = args.token or KeyringSecrets().get("client-tokens", args.client_id)
    if not token:
        print(
            f"No token for client {args.client_id!r} in the keyring. Set one with\n"
            f"  python -m glados.secrets set client-tokens {args.client_id}\n"
            "or pass --token for a dev fixture.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"=== GLaDOS bake-off -- slot {args.slot} ===")
    print(f"server: {args.url}   client: {args.client_id}/{args.room}")
    print(f"cart reset: {'ON (--reset) -- single-milk reseed before stateful block' if args.reset else 'OFF -- stateful tests share an unreset cart'}")
    print("Model under test is the server's configured [llm] model -- confirm it "
          "matches the slot.\n")

    # Dunnes search_results frames run >1 MB (30 full product records), past
    # the websockets client default max_size. The browser client has no such
    # cap; lift it here so the runner doesn't 1009-close mid-suite.
    async with websockets.connect(
        args.url, max_size=16 * 1024 * 1024, ssl=ssl_context(args.url)
    ) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "client_id": args.client_id,
            "room_id": args.room,
            "role": "ui",
            "token": token,
        }))
        only = {t.strip().upper() for t in args.only.split(",")} if args.only else None
        reset_pending = args.reset
        for test in TESTS:
            if only is not None and test.id not in only:
                continue
            if reset_pending and test.stateful:
                await _reset_cart(ws)
                reset_pending = False
            note = "  (memory-dependent -- v0 single-turn may not honour this)" if test.memory_dependent else ""
            print(f"--- {test.id}{note}")
            print(f"    pass if: {test.criterion}")
            for prompt in test.prompts:
                rep = await _run_turn(ws, prompt)
                _print_turn(rep)
            print()
    print("Done. Fill in the 0/1/2 scorecard in MODEL_BAKE_OFF.md from the above.")


def main() -> None:
    # Replies carry EUR and other non-cp1252 glyphs; the Windows console's
    # default codec raises UnicodeEncodeError on them. Force UTF-8 with
    # replacement so a stray glyph never aborts the run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="GLaDOS model bake-off runner")
    p.add_argument("--slot", default="?", help="label for this run (A/B/C) -- does not change the model")
    p.add_argument("--url", default="wss://127.0.0.1:8765/ws/v1")
    p.add_argument("--client-id", default="desk-ui")
    p.add_argument("--room", default="desk")
    p.add_argument("--token", default=None, help="override the keyring token (dev fixture)")
    p.add_argument("--only", default=None, help="comma-separated test ids to run (e.g. T9,T1) -- runs one at a time")
    p.add_argument("--reset", action="store_true", help="empty + reseed a single-milk cart before the stateful block (LLM-mediated via user_text; clears the persisted server-side cart)")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
