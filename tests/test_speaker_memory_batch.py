"""The nightly memory batch (DESIGN-speaker-memory.md, "Tests owed").

Every case here is about one promise: a staged note is consumed exactly once,
in the transaction that writes what it produced, whatever cancels, crashes,
forgets or reboots around it -- and the batch never runs twice for a day nor
outgrows the day's caps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime

import pytest

from glados.core.memory_wiring import open_speaker_memory
from glados.core.secrets import InMemorySecrets
from glados.core.speaker_memory import EXPIRED_COUNTER, SpeakerMemoryStore, notes_cipher

from .speaker_memory_helpers import (
    GROUP,
    KEY,
    BlockingLLM,
    Clock,
    ScriptedLLM,
    adds_reply,
    make_store,
    make_writer,
    run_idle,
    settle,
    stage,
)

VEG = adds_reply("Is vegetarian.")


@pytest.fixture
def store(tmp_path):
    s = make_store(tmp_path / "mem.sqlite")
    yield s
    s.close()


def _texts(store, key=KEY):
    return sorted(f.text for f in store.facts(*key))


def _note_ids(prompt):
    import re

    return [int(i) for i in re.findall(r'<note id="(\d+)">', prompt[1].content)]


# ---- the gate -------------------------------------------------------------


async def test_a_batch_waits_for_ten_idle_minutes_then_runs_and_rewarms(store):
    clock = Clock()
    rewarmed = []

    async def rewarm():
        rewarmed.append(True)

    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, rewarm=rewarm)
    stage(store, "I'm vegetarian")
    task = asyncio.create_task(writer.run())
    clock.now += 599
    await settle(writer)
    assert llm.prompts == []
    task.cancel()
    await run_idle(writer, clock)
    assert _texts(store) == ["Is vegetarian."] and rewarmed == [True]
    assert store.staged_count() == 0 and store.ran_on("2026-10-18")


@pytest.mark.parametrize("flag", ["busy", "warm"])
async def test_no_batch_while_a_turn_runs_or_the_model_is_cold(store, flag):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, flags = make_writer(store, llm, clock=clock)
    flags[flag] = not flags[flag]
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert llm.prompts == [] and store.staged_count() == 1


async def test_a_turn_resets_the_idle_window(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock)
    stage(store, "I'm vegetarian")
    clock.now += 500
    writer.turn_starting()
    clock.now += 500
    task = asyncio.create_task(writer.run())
    await settle(writer)
    task.cancel()
    assert llm.prompts == []


async def test_dst_fall_back_runs_the_date_once(store):
    """25-10-2026 01:30 happens twice in Ireland; the stored date does not."""
    clock = Clock()
    llm = ScriptedLLM(VEG)
    fall_back = datetime(2026, 10, 25, 1, 30)
    writer, _ = make_writer(store, llm, clock=clock, local_now=lambda: fall_back)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    clock.now -= 3600
    stage(store, "my daughter is Ema")
    await run_idle(writer, clock)
    assert len(llm.prompts) == 1 and store.staged_count() == 1


async def test_a_batch_window_only_narrows_when_it_may_run(store):
    from datetime import time as t

    clock = Clock()
    llm = ScriptedLLM(VEG)
    now = {"at": datetime(2026, 10, 18, 14, 0)}
    writer, _ = make_writer(store, llm, clock=clock, local_now=lambda: now["at"],
                            window=(t(23, 0), t(6, 0)))
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert llm.prompts == []
    now["at"] = datetime(2026, 10, 18, 23, 30)
    await run_idle(writer, clock)
    assert _texts(store) == ["Is vegetarian."]


async def test_without_the_notes_key_no_batch_runs(tmp_path):
    db = tmp_path / "mem.sqlite"
    seeded = make_store(db)
    stage(seeded, "I'm vegetarian")
    seeded.close()
    locked = make_store(db, cipher=None)
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(locked, llm, clock=clock)
    await run_idle(writer, clock)
    assert llm.prompts == [] and locked.staged_count() == 1
    locked.close()


# ---- inserts and preemption -------------------------------------------------


async def test_a_note_inserted_during_a_run_survives_the_chunk_it_missed(store):
    clock = Clock()
    staged_mid_call = []

    def reply(messages):
        if not staged_mid_call:
            stage(store, "my daughter is Ema")
            staged_mid_call.append(True)
        return VEG(messages)

    llm = ScriptedLLM(reply)
    writer, _ = make_writer(store, llm, clock=clock)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    first, second = (_note_ids(p) for p in llm.prompts)
    assert len(first) == 1 and len(second) == 1 and first != second
    assert store.staged_count() == 0


async def test_preempt_mid_call_keeps_the_notes_and_the_next_window_applies_once(store):
    clock = Clock()
    llm = BlockingLLM()
    writer, _ = make_writer(store, llm, clock=clock)
    stage(store, "I'm vegetarian")
    clock.now += 601
    task = asyncio.create_task(writer.run())
    writer.poke()
    await asyncio.wait_for(llm.started.wait(), 1)
    writer.turn_starting()
    await asyncio.sleep(0.05)
    task.cancel()
    assert llm.cancelled and writer.chunks_preempted == 1
    assert store.staged_count() == 1 and not store.ran_on("2026-10-18")
    writer._llm = ScriptedLLM(VEG)
    await run_idle(writer, clock)
    assert _texts(store) == ["Is vegetarian."] and store.staged_count() == 0


async def test_preempt_mid_apply_commits_once_and_never_reapplies(store, monkeypatch):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock)
    real_apply = store.apply_chunk_sync

    async def preempting_apply(write, enrolled):
        writer.turn_starting()
        await asyncio.sleep(0.01)
        return await asyncio.to_thread(real_apply, write, enrolled)

    monkeypatch.setattr(store, "apply_chunk", preempting_apply)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert writer.chunks_preempted == 1
    await run_idle(writer, clock)
    assert _texts(store) == ["Is vegetarian."] and len(llm.prompts) == 1


async def test_preempt_between_chunks_resumes_with_the_rest(store):
    clock = Clock()
    holder = {}

    async def rewarm():
        holder["writer"].turn_starting()
        await asyncio.sleep(0)

    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, rewarm=rewarm)
    holder["writer"] = writer
    stage(store, "I'm vegetarian", group=(*KEY, "s1"))
    stage(store, "I'm vegetarian", group=(*KEY, "s2"))
    await run_idle(writer, clock)
    assert len(llm.prompts) == 1 and store.staged_count() == 1
    assert not store.ran_on("2026-10-18")
    await run_idle(writer, clock)
    assert len(llm.prompts) == 2 and store.staged_count() == 0
    assert _texts(store) == ["Is vegetarian."]


async def test_shutdown_mid_apply_waits_for_the_commit(store, monkeypatch):
    clock = Clock()
    entered, release = threading.Event(), threading.Event()
    real_apply = store.apply_chunk_sync

    def held_apply(*args):
        entered.set()
        release.wait(5)
        return real_apply(*args)

    monkeypatch.setattr(store, "apply_chunk_sync", held_apply)
    writer, _ = make_writer(store, ScriptedLLM(VEG), clock=clock)
    stage(store, "I'm vegetarian")
    clock.now += 601
    task = asyncio.create_task(writer.run())
    writer.poke()
    await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(writer.close())
    await asyncio.sleep(0.05)
    assert not closing.done(), "close must wait for the threaded apply"
    release.set()
    await asyncio.wait_for(closing, 2)
    task.cancel()
    assert _texts(store) == ["Is vegetarian."] and store.staged_count() == 0


async def test_crash_then_catch_up_gives_the_same_facts(tmp_path):
    from cryptography.fernet import Fernet

    cipher = Fernet(Fernet.generate_key())
    said = ["I'm vegetarian", "my daughter is Ema"]

    async def full_run(db, *, crash_first: bool):
        store = make_store(db, cipher=cipher)
        for text in said:
            stage(store, text)
        if crash_first:
            clock = Clock()
            llm = BlockingLLM()
            writer, _ = make_writer(store, llm, clock=clock)
            clock.now += 601
            task = asyncio.create_task(writer.run())
            writer.poke()
            await asyncio.wait_for(llm.started.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            store.close()
            store = make_store(db, cipher=cipher)
        clock = Clock()
        writer, _ = make_writer(store, ScriptedLLM(adds_reply("Is vegetarian.", "Has a daughter called Ema.")), clock=clock)
        await run_idle(writer, clock)
        facts, left = _texts(store), store.staged_count()
        store.close()
        return facts, left

    crashed = await full_run(tmp_path / "a.sqlite", crash_first=True)
    clean = await full_run(tmp_path / "b.sqlite", crash_first=False)
    assert crashed == clean == (["Has a daughter called Ema.", "Is vegetarian."], 0)


# ---- forgets, staleness, enrollment ------------------------------------------


async def test_forget_session_mid_run_deletes_its_notes_and_writes_nothing(store, tmp_path):
    clock = Clock()

    def reply(messages):
        cli = SpeakerMemoryStore(tmp_path / "mem.sqlite")
        cli.forget_session("s1")
        cli.close()
        return VEG(messages)

    writer, _ = make_writer(store, ScriptedLLM(reply), clock=clock)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert _texts(store) == [] and store.staged_count() == 0


async def test_a_fact_forgotten_mid_call_retries_the_same_notes(store, tmp_path):
    from .speaker_memory_helpers import seed

    seed(store, text="Has a cat.")
    fact_id = store.facts(*KEY)[0].id
    clock = Clock()
    calls = []

    def reply(messages):
        calls.append(_note_ids(messages))
        if len(calls) == 1:
            cli = SpeakerMemoryStore(tmp_path / "mem.sqlite")
            cli.forget_fact(fact_id)
            cli.close()
        return VEG(messages)

    writer, _ = make_writer(store, ScriptedLLM(reply), clock=clock)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert _texts(store) == ["Is vegetarian."]


async def test_forget_fact_also_deletes_a_staged_note_it_cites(store):
    stage(store, "I'm vegetarian")
    chunk = store.read_chunk(GROUP, 10)
    from glados.core.memory_facts import AcceptedAdd
    from glados.core.speaker_memory import ChunkWrite

    store.apply_chunk_sync(
        ChunkWrite(group=GROUP, run_date="d", note_ids=(), expected=store.stamp(*KEY),
                   adds=(AcceptedAdd("Is vegetarian.", chunk.ids, False),)),
        lambda p: True,
    )
    assert store.staged_count() == 1
    removed = store.forget_fact(store.facts(*KEY)[0].id)
    assert (removed.facts, removed.notes) == (1, 1)


async def test_forget_session_removes_a_fact_any_of_its_sessions_backed(store):
    clock = Clock()
    writer, _ = make_writer(store, ScriptedLLM(VEG), clock=clock)
    stage(store, "I'm vegetarian", group=(*KEY, "s1"))
    stage(store, "still vegetarian", group=(*KEY, "s2"))
    await run_idle(writer, clock)
    assert _texts(store) == ["Is vegetarian."]
    assert store.forget_session("s2").facts == 1
    assert _texts(store) == []


async def test_wipe_reaches_staged_notes(store):
    stage(store, "I'm vegetarian")
    stage(store, "other room", group=("livingroom", "qcko", "s9"))
    removed = store.wipe(*KEY)
    assert removed.notes == 1 and store.staged_count() == 1


async def test_notes_of_a_person_unenrolled_overnight_are_purged_not_written(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, enrolled=lambda person: False)
    stage(store, "I'm vegetarian")
    stage(store, "elsewhere", group=("livingroom", "qcko", "s2"))
    await run_idle(writer, clock)
    assert _texts(store) == [] and store.staged_count() == 0


# ---- chunking and caps ------------------------------------------------------


async def test_a_chunk_over_the_window_is_split_before_it_is_sent(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, context_tokens=1100, output_tokens=100)
    for i in range(6):
        stage(store, f"note {i} " + "words " * 60)
    await run_idle(writer, clock, rounds=60)
    sizes = [len(_note_ids(p)) for p in llm.prompts]
    assert len(sizes) > 1 and sum(sizes) == 6
    assert all(sum(len(m.content) for m in p) // 2 <= 1000 for p in llm.prompts)
    assert store.staged_count() == 0


async def test_one_note_too_big_for_the_window_is_consumed_unsent(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, context_tokens=600, output_tokens=100)
    stage(store, "words " * 400)
    await run_idle(writer, clock)
    assert llm.prompts == [] and store.staged_count() == 0


async def test_chunks_hold_at_most_max_notes_per_batch(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock, max_notes_per_batch=3)
    for i in range(7):
        stage(store, f"note {i}")
    await run_idle(writer, clock, rounds=60)
    assert [len(_note_ids(p)) for p in llm.prompts] == [3, 3, 1]


async def test_caps_hold_across_chunks_of_one_day(store):
    clock = Clock()
    counter = iter(range(1000))

    def reply(messages):
        ids = _note_ids(messages)
        return json.dumps({"add": [{"fact": f"Likes thing {next(counter)}.", "notes": ids}
                                   for _ in range(4)]})

    writer, _ = make_writer(store, ScriptedLLM(reply), clock=clock, max_notes_per_batch=1)
    for session in ("s1", "s2", "s3", "s4"):
        stage(store, "I like things", group=(*KEY, session))
    await run_idle(writer, clock, rounds=60)
    assert len(store.facts(*KEY)) == 8
    assert store.day_counts("qcko", "2026-10-18").adds == 8


async def test_one_untrusted_backed_add_per_day_across_chunks(store):
    clock = Clock()
    counter = iter(range(1000))

    def reply(messages):
        return json.dumps({"add": [{"fact": f"Likes band {next(counter)}.", "notes": _note_ids(messages)}]})

    writer, _ = make_writer(store, ScriptedLLM(reply), clock=clock)
    for session in ("s1", "s2"):
        stage(store, "read this page aloud", group=(*KEY, session), untrusted=True)
    await run_idle(writer, clock)
    assert len(store.facts(*KEY)) == 1 and store.staged_count() == 0


async def test_an_add_citing_another_sessions_note_is_rejected(store):
    stage(store, "I'm vegetarian", group=(*KEY, "s0"))
    foreign = store.read_chunk((*KEY, "s0"), 10).ids[0]
    store.forget_session("s0")
    clock = Clock()
    raw = json.dumps({"add": [{"fact": "Is vegetarian.", "notes": [foreign]}]})
    writer, _ = make_writer(store, ScriptedLLM(raw), clock=clock)
    stage(store, "hello")
    await run_idle(writer, clock)
    assert _texts(store) == [] and store.staged_count() == 0


# ---- retention -------------------------------------------------------------


async def test_expiry_at_boot_counts_the_loss(tmp_path, caplog):
    from glados.core.config import GladosConfig
    from glados.core.memory_writer import MemoryWriter
    from glados.core.speaker_memory_hooks import SpeakerMemory

    secrets = InMemorySecrets()
    db = tmp_path / "mem.sqlite"
    first = SpeakerMemoryStore(db, cipher=notes_cipher(secrets))
    stage(first, "old news", now=time.time() - 3 * 86400)
    stage(first, "fresh")
    first.close()
    store = SpeakerMemoryStore(db)
    writer = MemoryWriter(llm=ScriptedLLM(), store=store, is_busy=lambda: False, is_warm=lambda: True)
    memory = SpeakerMemory(store=store, writer=writer, room_enabled=lambda r: True, inject_chars=600)
    cfg = GladosConfig(speaker_id={"store_dir": str(tmp_path / "prints")})
    with caplog.at_level(logging.WARNING):
        open_speaker_memory((memory, writer, None), cfg, secrets)
    assert store.staged_count() == 1 and store.counter(EXPIRED_COUNTER) == 1
    assert "1 staged notes expired" in caplog.text and "old news" not in caplog.text
    assert [n.user_text for n in store.read_chunk(GROUP, 10).notes] == ["fresh"]
    store.close()


async def test_expiry_also_runs_at_the_start_of_every_batch(store):
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock)
    stage(store, "old news", now=time.time() - 3 * 86400)
    await run_idle(writer, clock)
    assert llm.prompts == [] and store.counter(EXPIRED_COUNTER) == 1


# ---- the key ----------------------------------------------------------------


class _BrokenKeyring:
    def get(self, scope, name):
        raise RuntimeError("no backend")

    def set(self, scope, name, value):
        raise RuntimeError("no backend")


class _ForgetfulKeyring:
    def get(self, scope, name):
        return None

    def set(self, scope, name, value):
        pass


@pytest.mark.parametrize("secrets", [_BrokenKeyring(), _ForgetfulKeyring()], ids=["broken", "forgetful"])
async def test_a_missing_notes_key_stages_nothing(tmp_path, secrets, caplog):
    from glados.core.speaker import SpeakerIdentity
    from glados.core.speaker_memory_hooks import SpeakerMemory

    assert notes_cipher(secrets) is None
    store = SpeakerMemoryStore(tmp_path / "mem.sqlite", cipher=notes_cipher(secrets))
    writer, _ = make_writer(store, ScriptedLLM())
    memory = SpeakerMemory(store=store, writer=writer, room_enabled=lambda r: True, inject_chars=600)
    ema = SpeakerIdentity(person="ema", session_key="ema", source="voice", score=0.9)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            memory.on_turn_committed(room_id="kitchen", speaker=ema, session_id="s",
                                     user_text="I'm vegetarian", reply="ok", untrusted=False)
        await memory.drain()
    assert store.staged_count() == 0
    assert caplog.text.count("no notes key") == 1
    store.close()


def test_the_notes_key_is_created_once_and_reused():
    secrets = InMemorySecrets()
    first = notes_cipher(secrets)
    token = first.encrypt(b"x")
    assert notes_cipher(secrets).decrypt(token) == b"x"


async def test_a_note_under_a_lost_key_is_consumed_not_retried(tmp_path):
    db = tmp_path / "mem.sqlite"
    old = make_store(db)
    stage(old, "I'm vegetarian")
    old.close()
    store = make_store(db)
    clock = Clock()
    llm = ScriptedLLM(VEG)
    writer, _ = make_writer(store, llm, clock=clock)
    await run_idle(writer, clock)
    assert llm.prompts == [] and store.staged_count() == 0
    store.close()


# ---- privacy ----------------------------------------------------------------

SECRET = "my-secret-phrase-zebra"


@pytest.mark.parametrize("failure", ["bad-json", "sqlite", "llm-raises"])
async def test_a_failing_batch_logs_no_note_text(store, monkeypatch, caplog, failure):
    import sqlite3

    clock = Clock()
    llm = ScriptedLLM(f"not json {SECRET}")
    if failure == "llm-raises":
        def boom(messages):
            raise RuntimeError(f"model choked on {SECRET}")
        llm = ScriptedLLM(boom)
    if failure == "sqlite":
        llm = ScriptedLLM(adds_reply("Likes zebras."))

        def broken(*args):
            raise sqlite3.OperationalError(f"disk I/O near {SECRET}")

        monkeypatch.setattr(store, "apply_chunk_sync", broken)
    writer, _ = make_writer(store, llm, clock=clock)
    stage(store, f"I said {SECRET}", reply=f"you said {SECRET}")
    with caplog.at_level(logging.DEBUG):
        await run_idle(writer, clock)
    assert SECRET not in caplog.text
    assert "memory" in caplog.text


async def test_caps_are_per_person_across_rooms(store):
    clock = Clock()
    counter = iter(range(1000))

    def reply(messages):
        ids = _note_ids(messages)
        return json.dumps({"add": [{"fact": f"Likes thing {next(counter)}.", "notes": ids}
                                   for _ in range(6)]})

    writer, _ = make_writer(store, ScriptedLLM(reply), clock=clock)
    for room in ("kitchen", "livingroom"):
        stage(store, "I like things", group=(room, "qcko", f"s-{room}"))
    await run_idle(writer, clock, rounds=60)
    total = len(store.facts("kitchen", "qcko")) + len(store.facts("livingroom", "qcko"))
    assert total == 8


async def test_unknown_enrollment_keeps_the_notes(store):
    from glados.core.memory_writer import EnrollmentUnknown

    def unknown(person):
        raise EnrollmentUnknown("no voiceprint loadable")

    clock = Clock()
    writer, _ = make_writer(store, ScriptedLLM(VEG), clock=clock, enrolled=unknown)
    stage(store, "I'm vegetarian")
    await run_idle(writer, clock)
    assert store.staged_count() == 1 and _texts(store) == [] and writer.chunks_failed == 1


def test_no_loadable_voiceprint_reads_as_unknown_not_nobody(tmp_path):
    from glados.core.memory_wiring import _enrolled_check
    from glados.core.memory_writer import EnrollmentUnknown
    from glados.voiceid.store import VoiceprintStore

    check = _enrolled_check(VoiceprintStore(tmp_path / "prints", InMemorySecrets()))
    with pytest.raises(EnrollmentUnknown):
        check("qcko")
