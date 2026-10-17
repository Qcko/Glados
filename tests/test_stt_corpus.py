"""Pure-logic tests for scripts/stt_corpus.py (no microphone)."""

from __future__ import annotations

import json
import wave
from importlib import util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parent.parent / "scripts" / "stt_corpus.py"
    spec = util.spec_from_file_location("stt_corpus", path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corpus = _load()


def test_people_read_only_from_people_section(tmp_path: Path) -> None:
    lex = tmp_path / "lex.txt"
    lex.write_text("[terms]\nX = not a person\n[people]\n# PERSON_C = commented\nPERSON_A = Ann\n", encoding="utf-8")
    assert corpus.load_people(lex) == {"PERSON_A": "Ann"}


def test_missing_lexicon_means_no_people(tmp_path: Path) -> None:
    assert corpus.load_people(tmp_path / "absent.txt") == {}


def test_person_prompts_dropped_without_names_and_filled_with_them() -> None:
    without = corpus.load_items("read", {})
    assert not any("{PERSON" in item["show"] for item in without)
    with_names = corpus.load_items("read", {"PERSON_A": "Ann", "PERSON_B": "Bo"})
    p01 = next(item for item in with_names if item["id"] == "p01")
    assert p01["text"] == "Tell Ann dinner is ready"
    assert p01["entities"] == [["Ann", "lex"]]
    assert len(with_names) == len(without) + 4


def test_prompt_ids_unique_and_held_out_share_near_thirty_percent() -> None:
    items = corpus.load_items("read", {"PERSON_A": "A", "PERSON_B": "B"})
    assert len({item["id"] for item in items}) == len(items)
    held = sum(any(tag == "held" for _, tag in item["entities"]) for item in items)
    assert 0.25 <= held / len(items) <= 0.4


def test_held_terms_absent_from_lexicon() -> None:
    lexicon = (corpus.INPUTS / "lexicon.example.txt").read_text(encoding="utf-8").lower().splitlines()
    held = {term.lower() for item in corpus.load_items("read", {}) for term, tag in item["entities"] if tag == "held"}
    assert not held & set(lexicon)


@pytest.mark.parametrize("kind,count", [("scenario", 15), ("nospeech", 30)])
def test_unscripted_kinds_have_empty_reference(kind: str, count: int) -> None:
    items = corpus.load_items(kind, {})
    assert len(items) == count
    assert all(item["text"] == "" for item in items)


def test_save_then_resume_skips_only_same_speaker_and_mic(tmp_path: Path) -> None:
    items = corpus.load_items("scenario", {})
    record = corpus.save_clip(tmp_path, "spk1", "desk", "scenario", items[0], b"\x01\x00" * 160)
    with wave.open(str(tmp_path / record["audio"]), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()) == (16_000, 1, 2, 160)
    manifest = tmp_path / "manifest.jsonl"
    assert json.loads(manifest.read_text(encoding="utf-8"))["speaker"] == "spk1"
    assert len(corpus.remaining(items, manifest, "spk1", "desk")) == len(items) - 1
    assert len(corpus.remaining(items, manifest, "spk2", "desk")) == len(items)
    assert len(corpus.remaining(items, manifest, "spk1", "kitchen")) == len(items)


def test_truncated_manifest_line_does_not_break_resume(tmp_path: Path) -> None:
    items = corpus.load_items("scenario", {})
    corpus.save_clip(tmp_path, "spk1", "desk", "scenario", items[0], b"\x00\x00")
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "sc02", "spea')
    assert len(corpus.remaining(items, manifest, "spk1", "desk")) == len(items) - 1
