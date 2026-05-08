"""Tiny smoke client: hello + user_text, print everything the server sends."""

import asyncio
import json
import sys

import websockets


async def turn(client_id: str, room_id: str, token: str, text: str) -> None:
    uri = "ws://127.0.0.1:8765/ws/v1"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "client_id": client_id,
            "room_id": room_id,
            "role": "ui",
            "token": token,
        }))
        await ws.send(json.dumps({"type": "user_text", "text": text}))
        for _ in range(3):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            print(f"[{client_id}] <- {msg}")


async def main() -> None:
    print("== happy path: desk-ui ==")
    await turn("desk-ui", "desk", "dev-token-desk", "hello glados")

    print("\n== second room: desk2-ui ==")
    await turn("desk2-ui", "desk2", "dev-token-desk2", "different room")

    print("\n== bad token ==")
    await turn("desk-ui", "desk", "wrong", "should fail")

    print("\n== binding mismatch (desk-ui claiming desk2) ==")
    try:
        await turn("desk-ui", "desk2", "dev-token-desk", "should fail")
    except Exception as e:
        print(f"closed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
