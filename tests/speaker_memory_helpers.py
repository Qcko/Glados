"""Shared fakes for the per-speaker memory tests."""

from __future__ import annotations

import asyncio
from datetime import datetime

from cryptography.fernet import Fernet

from glados.core.adapters import LLMMessage, LLMText
from glados.core.memory_facts import AcceptedAdd, StagedNote
from glados.core.memory_writer import BatchPolicy, MemoryWriter
from glados.core.speaker_memory import ChunkWrite, SpeakerMemoryStore

KEY = ("kitchen", "qcko")
GROUP = (*KEY, "s1")
TODAY = datetime(2026, 10, 18, 3, 0)


class ScriptedLLM:
    """Replies with fixed text (or a callable of the prompt) and records
    every prompt it was sent."""

    is_local = True

    def __init__(self, reply="ok") -> None:
        self.reply = reply
        self.prompts: list[list[LLMMessage]] = []

    async def chat(self, messages, tools):
        self.prompts.append(list(messages))
        text = self.reply(messages) if callable(self.reply) else self.reply
        yield LLMText(text=text)


class BlockingLLM:
    """Never answers until cancelled; records that it was cancelled."""

    is_local = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def chat(self, messages, tools):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield LLMText(text="never")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_store(path, *, max_facts: int = 30, cipher: Fernet | None | str = "new"):
    key = Fernet(Fernet.generate_key()) if cipher == "new" else cipher
    return SpeakerMemoryStore(path, max_facts=max_facts, cipher=key)


def stage(store, text, *, group=GROUP, reply=None, untrusted=False, now=None) -> None:
    note = StagedNote(id=0, user_text=text, reply=reply, untrusted=untrusted)
    assert store.stage_note(group, f"u-{text}", note, now=now)


def seed(store, key=KEY, text="Is vegetarian.", session="s") -> None:
    """A fact as if an earlier run had written it."""
    store.apply_chunk_sync(
        ChunkWrite(
            group=(*key, session), run_date="2026-01-01", note_ids=(),
            expected=store.stamp(*key),
            adds=(AcceptedAdd(text=text, note_ids=(), untrusted=False),),
        ),
        lambda person: True,
    )


def adds_reply(*facts: str):
    """A writer reply that cites every note it was shown for each fact."""
    import json
    import re

    def reply(messages):
        ids = [int(i) for i in re.findall(r'<note id="(\d+)">', messages[1].content)]
        return json.dumps({"add": [{"fact": f, "notes": ids} for f in facts], "drop": []})

    return reply


def make_writer(store, llm, *, clock=None, rewarm=None, enrolled=lambda p: True,
                local_now=lambda: TODAY, **policy):
    flags = {"busy": False, "warm": True}
    writer = MemoryWriter(
        llm=llm, store=store,
        is_busy=lambda: flags["busy"], is_warm=lambda: flags["warm"],
        rewarm=rewarm, is_enrolled=enrolled,
        policy=BatchPolicy(**policy), poll_s=0.01,
        clock=clock or Clock(), local_now=local_now,
    )
    return writer, flags


async def settle(writer, rounds: int = 30) -> None:
    for _ in range(rounds):
        writer.poke()
        await asyncio.sleep(0.01)


async def run_idle(writer, clock, rounds: int = 30) -> None:
    """Let the house go idle past the gate and run the loop for a while."""
    clock.now += 601
    task = asyncio.create_task(writer.run())
    await settle(writer, rounds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
