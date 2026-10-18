"""Speaker identity (DESIGN-speaker-id.md): the published contract, the
decision rule, the encrypted store, the identifier's failure modes, the
pipeline pairing/ordering, and how the Organizer keys sessions on it."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from glados.audio.pipeline import AudioPipeline
from glados.audio.stt.fake import FakeSTT
from glados.audio.vad.fake import FakeVAD
from glados.core.adapters import LLMText
from glados.core.config import ClientBinding, SpeakerIdConfig
from glados.core.organizer import Organizer
from glados.core.secrets import InMemorySecrets
from glados.core.sessions import SessionRegistry
from glados.core.speaker import (
    NO_IDENTIFIER,
    SpeakerIdentity,
    SpeakerMatch,
    Utterance,
    resolve_identity,
)
from glados.core.traces import TraceStore
from glados.mcp.registry import MCPRegistry
from glados.voiceid import build_identifier
from glados.voiceid.__main__ import main as voiceid_cli
from glados.voiceid.embedder import FakeEmbedder, unit
from glados.voiceid.identifier import SpeakerIdentifier, decide
from glados.voiceid.store import StoreError, Voiceprint, VoiceprintStore

ALICE = [1.0, 0.0, 0.0, 0.0]
BOB = [0.0, 1.0, 0.0, 0.0]
ONE_SECOND = 16_000


def _pcm(n_samples: int, value: int = 1) -> bytes:
    return struct.pack("<h", value) * n_samples


def _frame(seq: int, pcm: bytes) -> bytes:
    return struct.pack(">I", seq) + pcm


def _print(person: str, vector, model_id: str = "fake-v1") -> Voiceprint:
    return Voiceprint(person=person, model_id=model_id, centroid=unit(vector))


# ---- contract ------------------------------------------------------------


def test_typed_turn_is_the_bindings_user_but_unverified() -> None:
    identity = resolve_identity(None, "qcko")
    assert identity == SpeakerIdentity(person="qcko", session_key="qcko", source="typed")
    assert not identity.verified


def test_unidentified_voice_names_nobody_but_keeps_the_room_session() -> None:
    identity = resolve_identity(SpeakerMatch.none("low_score", 0.4), "qcko")
    assert identity.person is None
    assert identity.session_key == "qcko"
    assert not identity.verified


def test_identified_voice_is_verified_and_keyed_on_the_person() -> None:
    identity = resolve_identity(SpeakerMatch("alice", 0.91, "accepted"), "qcko")
    assert identity == SpeakerIdentity("alice", "alice", "voice", 0.91)
    assert identity.verified


def test_a_voiceprint_never_prints_its_vector() -> None:
    assert "centroid" not in repr(_print("alice", ALICE))


# ---- decision rule -------------------------------------------------------


def test_decide_accepts_exactly_at_threshold_and_margin() -> None:
    prints = (_print("alice", ALICE), _print("bob", BOB))
    match = decide(unit(ALICE), prints, accept_threshold=1.0, margin=1.0)
    assert (match.person, match.reason) == ("alice", "accepted")


def test_decide_below_threshold_names_nobody() -> None:
    probe = unit([0.7, 0.0, 0.714, 0.0])
    match = decide(probe, (_print("alice", ALICE),), accept_threshold=0.75, margin=0.1)
    assert match.person is None and match.reason == "low_score"


def test_decide_refuses_two_close_candidates() -> None:
    probe = unit([0.75, 0.66, 0.0, 0.0])
    prints = (_print("alice", ALICE), _print("bob", BOB))
    match = decide(probe, prints, accept_threshold=0.5, margin=0.2)
    assert match.person is None and match.reason == "low_margin"


def test_decide_with_one_person_measures_margin_from_zero() -> None:
    match = decide(unit(ALICE), (_print("alice", ALICE),), accept_threshold=0.7, margin=0.1)
    assert match.person == "alice"


# ---- store ---------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> VoiceprintStore:
    return VoiceprintStore(tmp_path / "prints", InMemorySecrets())


def test_store_round_trips_encrypted(store: VoiceprintStore, tmp_path: Path) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    raw = (tmp_path / "prints" / "alice.vp").read_bytes()
    assert b"alice" not in raw and b"centroid" not in raw
    (only,) = store.snapshot("fake-v1")
    assert only.person == "alice"
    assert np.allclose(only.centroid, unit(ALICE))


def test_store_refuses_to_overwrite_without_replace(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    with pytest.raises(StoreError, match="--replace"):
        store.enroll("alice", "fake-v1", [np.array(BOB)])
    store.enroll("alice", "fake-v1", [np.array(BOB)], replace=True)
    assert np.allclose(store.snapshot("fake-v1")[0].centroid, unit(BOB))


def test_forget_takes_effect_on_the_next_snapshot(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    assert store.snapshot("fake-v1")
    assert store.forget("alice")
    assert store.snapshot("fake-v1") == ()
    assert not store.forget("alice")


def test_a_corrupt_print_is_skipped_and_the_rest_load(
    store: VoiceprintStore, tmp_path: Path
) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    (tmp_path / "prints" / "bob.vp").write_bytes(b"not a fernet token")
    assert [p.person for p in store.snapshot("fake-v1")] == ["alice"]


def test_a_print_from_another_model_is_never_compared(
    store: VoiceprintStore, caplog: pytest.LogCaptureFixture
) -> None:
    store.enroll("alice", "old-model", [np.array(ALICE)])
    with caplog.at_level("WARNING"):
        assert store.snapshot("fake-v1") == ()
    assert any("re-enroll" in r.message for r in caplog.records)


def test_prints_without_the_keyring_key_load_nothing(tmp_path: Path) -> None:
    VoiceprintStore(tmp_path, InMemorySecrets()).enroll("alice", "m", [np.array(ALICE)])
    assert VoiceprintStore(tmp_path, InMemorySecrets()).snapshot("m") == ()


@pytest.mark.parametrize("person", ["", "Alice", "../evil", "a" * 33, "a b"])
def test_person_ids_are_validated(store: VoiceprintStore, person: str) -> None:
    with pytest.raises(StoreError):
        store.enroll(person, "m", [np.array(ALICE)])


# ---- identifier ----------------------------------------------------------


def _identifier(store, embedder, *, timeout_s: float = 0.5) -> SpeakerIdentifier:
    return SpeakerIdentifier(
        embedder, store, accept_threshold=0.7, margin=0.1, min_speech_s=1.0, timeout_s=timeout_s
    )


async def test_no_prints_embeds_nothing(store: VoiceprintStore) -> None:
    embedder = FakeEmbedder()
    match = await _identifier(store, embedder).identify(_pcm(ONE_SECOND))
    assert match.reason == "no_prints" and embedder.calls == 0


async def test_too_short_is_not_embedded(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    embedder = FakeEmbedder()
    match = await _identifier(store, embedder).identify(_pcm(ONE_SECOND - 1))
    assert match.reason == "too_short" and embedder.calls == 0


async def test_identifies_an_enrolled_voice(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    store.enroll("bob", "fake-v1", [np.array(BOB)])
    clip = _pcm(ONE_SECOND, 7)
    match = await _identifier(store, FakeEmbedder({clip: BOB})).identify(clip)
    assert (match.person, match.reason) == ("bob", "accepted")


async def test_a_stranger_is_not_identified(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    match = await _identifier(store, FakeEmbedder({}, dim=4)).identify(_pcm(ONE_SECOND, 3))
    assert match.person is None


async def test_slow_embed_times_out_and_the_next_one_is_busy(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    ident = _identifier(store, FakeEmbedder(delay_s=0.3), timeout_s=0.05)
    try:
        started = time.monotonic()
        assert (await ident.identify(_pcm(ONE_SECOND))).reason == "timeout"
        assert time.monotonic() - started < 0.25
        assert (await ident.identify(_pcm(ONE_SECOND))).reason == "busy"
        await asyncio.sleep(0.35)
        assert (await ident.identify(_pcm(ONE_SECOND))).reason != "busy"
    finally:
        ident.close()


async def test_an_embed_error_names_nobody(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    ident = _identifier(store, FakeEmbedder(error=RuntimeError("onnx")))
    assert (await ident.identify(_pcm(ONE_SECOND))).reason == "error"


def test_backend_off_builds_no_identifier(tmp_path: Path) -> None:
    cfg = SpeakerIdConfig(store_dir=tmp_path)
    assert build_identifier(cfg, InMemorySecrets()) is None
    assert build_identifier(cfg.model_copy(update={"backend": "fake"}), InMemorySecrets())


# ---- pipeline ------------------------------------------------------------


class _VoiceByPcm:
    """Identifier stand-in: the pcm's first sample names the speaker, and
    `delays` slows chosen speakers down."""

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self._delays = delays or {}

    async def identify(self, pcm: bytes) -> SpeakerMatch:
        person = {1: "alice", 2: "bob"}[struct.unpack("<h", pcm[:2])[0]]
        await asyncio.sleep(self._delays.get(person, 0.0))
        return SpeakerMatch(person, 0.9, "accepted")


class _TextByPcm:
    def __init__(self, delays: dict[int, float] | None = None) -> None:
        self._delays = delays or {}

    async def transcribe(self, pcm: bytes) -> str:
        first = struct.unpack("<h", pcm[:2])[0]
        await asyncio.sleep(self._delays.get(first, 0.0))
        return f"utterance {first}"


def _pipeline(seen: list[Utterance], stt, identifier) -> AudioPipeline:
    async def on_utterance(u: Utterance) -> None:
        seen.append(u)

    return AudioPipeline(
        sink=None,
        vad=FakeVAD(utterance_samples=4),
        stt=stt,
        on_utterance=on_utterance,
        identifier=identifier,
    )


async def test_each_utterance_carries_its_own_speaker() -> None:
    seen: list[Utterance] = []
    pipe = _pipeline(seen, _TextByPcm(), _VoiceByPcm(delays={"alice": 0.05}))
    await pipe.feed_frame(_frame(0, _pcm(4, 1) + _pcm(4, 2)))
    await pipe.drain()
    assert [(u.text, u.speaker.person) for u in seen] == [
        ("utterance 1", "alice"),
        ("utterance 2", "bob"),
    ]


async def test_dispatch_keeps_vad_order_when_the_first_is_slower() -> None:
    seen: list[Utterance] = []
    pipe = _pipeline(seen, _TextByPcm(delays={1: 0.08}), _VoiceByPcm())
    await pipe.feed_frame(_frame(0, _pcm(4, 1) + _pcm(4, 2)))
    await pipe.drain()
    assert [u.text for u in seen] == ["utterance 1", "utterance 2"]


async def test_no_identifier_means_unidentified_voice() -> None:
    seen: list[Utterance] = []
    pipe = _pipeline(seen, FakeSTT("hi"), None)
    await pipe.feed_frame(_frame(0, _pcm(4)))
    await pipe.drain()
    assert seen[0].speaker == NO_IDENTIFIER


async def test_a_failed_transcription_does_not_block_the_next() -> None:
    seen: list[Utterance] = []

    class FirstFails:
        async def transcribe(self, pcm: bytes) -> str:
            if struct.unpack("<h", pcm[:2])[0] == 1:
                raise RuntimeError("boom")
            return "second"

    pipe = _pipeline(seen, FirstFails(), None)
    await pipe.feed_frame(_frame(0, _pcm(4, 1) + _pcm(4, 2)))
    await pipe.drain()
    assert [u.text for u in seen] == ["second"]


# ---- organizer -----------------------------------------------------------


class _SlowLLM:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def chat(self, messages, tools):
        yield LLMText(text="working ")
        self.entered.set()
        await asyncio.sleep(3600)


class _QuickLLM:
    async def chat(self, messages, tools):
        yield LLMText(text="ok")


MIC = ClientBinding(client_id="k-mic", room_id="kitchen", role="mic", default_user="qcko")
UI = ClientBinding(client_id="k-ui", room_id="kitchen", role="ui", default_user="qcko")


@asynccontextmanager
async def _organizer(tmp: Path, llm):
    sink: list[dict] = []

    async def send(client_id: str, msg: BaseModel) -> None:
        sink.append(msg.model_dump())

    bindings = {b.client_id: b for b in (MIC, UI)}
    org = Organizer(
        llm=llm,
        mcp=MCPRegistry(),
        traces=TraceStore(tmp),
        sessions=SessionRegistry(),
        send=send,
        binding_for_client=bindings.get,
        clients_in_room=lambda r: list(bindings),
    )
    try:
        yield org, sink
    finally:
        await org.close()


def _turn_starts(tmp: Path) -> list[dict]:
    events = []
    for path in sorted(tmp.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "turn_start":
                events.append(event)
    return events


async def test_identified_and_unidentified_voices_get_separate_sessions(tmp_path: Path) -> None:
    async with _organizer(tmp_path, _QuickLLM()) as (org, _):
        await org.handle_audio_text("k-mic", "hello", speaker=SpeakerMatch("alice", 0.9, "accepted"))
        await org.handle_audio_text("k-mic", "hello again", speaker=SpeakerMatch.none("too_short"))
        await org.handle_user_text("k-ui", "typed")
        await org.flush()
        alice = org.sessions.latest("kitchen", "alice")
        fallback = org.sessions.latest("kitchen", "qcko")
        assert alice is not None and fallback is not None
        assert alice.session_id != fallback.session_id
    starts = _turn_starts(tmp_path)
    assert {(e["speaker_person"], e["speaker_source"]) for e in starts} == {
        ("alice", "voice"),
        (None, "fallback"),
        ("qcko", "typed"),
    }
    assert all("centroid" not in json.dumps(e) for e in starts)


async def test_anyones_stop_cancels_the_rooms_turn(tmp_path: Path) -> None:
    llm = _SlowLLM()
    async with _organizer(tmp_path, llm) as (org, sink):
        await org.handle_audio_text("k-mic", "tell me a story", speaker=SpeakerMatch("alice", 0.9, "accepted"))
        await llm.entered.wait()
        await org.handle_audio_text("k-mic", "stop", speaker=SpeakerMatch("bob", 0.9, "accepted"))
        await org.flush()
        kinds = [m["type"] for m in sink]
        assert "cancelled" in kinds
        sessions = {m["session_id"] for m in sink if m["type"] == "welcome"}
        assert len(sessions) == 1, "bob's stop must not open his own turn"


# ---- enrollment CLI ------------------------------------------------------


def _wav(path: Path, value: int) -> Path:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(_pcm(ONE_SECOND, value))
    return path


def _config_dir(tmp: Path) -> Path:
    cfg = tmp / "cfg"
    cfg.mkdir()
    store = (tmp / "prints").as_posix()
    (cfg / "glados.toml").write_text(
        f'[speaker_id]\nbackend = "fake"\nstore_dir = "{store}"\n', encoding="utf-8"
    )
    (cfg / "rooms.toml").write_text(
        '[[clients]]\nclient_id = "k-mic"\nroom_id = "kitchen"\nrole = "mic"\n'
        'default_user = "qcko"\n',
        encoding="utf-8",
    )
    return cfg


def test_cli_enroll_list_forget(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cfg = _config_dir(tmp_path)
    secrets = InMemorySecrets()
    clips = [str(_wav(tmp_path / f"c{i}.wav", i + 1)) for i in range(3)]
    run = lambda *argv: voiceid_cli(["--config-dir", str(cfg), *argv], secrets=secrets)
    assert run("enroll", "stranger", *clips) == 1
    assert run("enroll", "qcko", *clips) == 0
    assert run("enroll", "qcko", *clips) == 1
    assert run("enroll", "qcko", "--replace", *clips) == 0
    assert run("list") == 0
    assert "qcko\tfake-v1" in capsys.readouterr().out
    assert run("forget", "qcko") == 0
    assert run("forget", "qcko") == 1


async def test_a_closed_identifier_names_nobody_and_is_not_stuck(store: VoiceprintStore) -> None:
    store.enroll("alice", "fake-v1", [np.array(ALICE)])
    ident = _identifier(store, FakeEmbedder())
    ident.close()
    assert (await ident.identify(_pcm(ONE_SECOND))).reason == "error"
    assert (await ident.identify(_pcm(ONE_SECOND))).reason == "error"


async def test_an_identify_crash_never_costs_the_transcript() -> None:
    class Crashing:
        async def identify(self, pcm: bytes) -> SpeakerMatch:
            raise RuntimeError("boom")

    seen: list[Utterance] = []
    pipe = _pipeline(seen, FakeSTT("still heard"), Crashing())
    await pipe.feed_frame(_frame(0, _pcm(4)))
    await pipe.drain()
    assert [(u.text, u.speaker.reason) for u in seen] == [("still heard", "error")]
