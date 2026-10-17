"""STT corpus recorder for the track 4 bake-off (DESIGN-stt-fit.md).

Shows one prompt at a time, records 16 kHz mono PCM16 from the default input
device, and appends a line per clip to `<out>/manifest.jsonl`. Re-running the
same speaker/mic/kind resumes where it stopped.

Usage:
    uv run --with sounddevice python scripts/stt_corpus.py \\
        --speaker spk1 --mic desk --out <lab-root>/glados/stt_corpus \\
        [--kind read|scenario|nospeech]

Keys per clip: Enter = start, Enter = stop, then Enter = keep, r = redo,
s = skip (offered again next run), q = quit. `scenario` clips have an empty `text` until someone types
the reference afterwards -- filter them out of a benchmark manifest until
then; `nospeech` clips have `text` = "" and are scored for hallucination.

Voice is biometric: keep `--out` outside the repo and label speakers spk1..n.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import wave
from pathlib import Path

RATE = 16_000
INPUTS = Path(__file__).resolve().parent.parent / "benchmarks" / "stt"
PERSON_SLOT = re.compile(r"\{(PERSON_[A-Z])\}")
NOSPEECH_CARDS = 30
TAIL_S = 0.3


def load_people(lexicon_path: Path) -> dict[str, str]:
    if not lexicon_path.exists():
        return {}
    people: dict[str, str] = {}
    in_people = False
    for raw in lexicon_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_people = line == "[people]"
        elif in_people and "=" in line and not line.startswith("#"):
            key, value = (part.strip() for part in line.split("=", 1))
            people[key] = value
    return people


def load_items(kind: str, people: dict[str, str]) -> list[dict]:
    if kind == "read":
        return _read_items(people)
    if kind == "scenario":
        return _scenario_items()
    return [{"id": f"n{i:02d}", "show": f"No speech #{i}: room tone / TV / kitchen noise", "text": "", "entities": []}
            for i in range(1, NOSPEECH_CARDS + 1)]


def _read_items(people: dict[str, str]) -> list[dict]:
    items = []
    for line in (INPUTS / "prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if PERSON_SLOT.search(record["text"]) and not _slots_filled(record["text"], people):
            continue
        text = fill_people(record["text"], people)
        entities = [[fill_people(term, people), tag] for term, tag in record["entities"]]
        items.append({"id": record["id"], "show": text, "text": text, "entities": entities})
    return items


def _slots_filled(text: str, people: dict[str, str]) -> bool:
    return all(slot in people for slot in PERSON_SLOT.findall(text))


def fill_people(text: str, people: dict[str, str]) -> str:
    return PERSON_SLOT.sub(lambda m: people.get(m.group(1), m.group(0)), text)


def _scenario_items() -> list[dict]:
    items = []
    for line in (INPUTS / "scenarios.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\d+)\.\s+(.*)", line.strip())
        if match:
            items.append({"id": f"sc{int(match.group(1)):02d}", "show": match.group(2), "text": "", "entities": []})
    return items


def remaining(items: list[dict], manifest_path: Path, speaker: str, mic: str) -> list[dict]:
    done = _recorded_ids(manifest_path, speaker, mic)
    return [item for item in items if item["id"] not in done]


def _recorded_ids(manifest_path: Path, speaker: str, mic: str) -> set[str]:
    if not manifest_path.exists():
        return set()
    done = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = _parse_line(line)
        if record and record.get("speaker") == speaker and record.get("mic") == mic:
            done.add(record["id"])
    return done


def _parse_line(line: str) -> dict | None:
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print(f"  warning: skipping unreadable manifest line: {line[:60]!r}")
        return None


def save_clip(out: Path, speaker: str, mic: str, kind: str, item: dict, pcm: bytes) -> dict:
    rel = Path(speaker) / mic / f"{item['id']}.wav"
    _write_wav(out / rel, pcm)
    record = {
        "id": item["id"], "audio": rel.as_posix(), "text": item["text"],
        "speaker": speaker, "mic": mic, "kind": kind, "entities": item["entities"],
    }
    with (out / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return record


def _write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm)


def record_until_enter() -> tuple[bytes, int]:
    """Record from the default input until Enter, plus a short tail so the last
    syllable is not cut. Returns (pcm, overflow_count)."""
    import sounddevice as sd

    chunks: list[bytes] = []
    overflows = [0]
    stop = threading.Event()

    def _callback(indata, _frames, _time, status) -> None:
        if status.input_overflow:
            overflows[0] += 1
        if not stop.is_set():
            chunks.append(bytes(indata))

    with _open_stream(sd, _callback):
        input("  SPEAK NOW... Enter to stop ")
        time.sleep(TAIL_S)
        stop.set()
    return b"".join(chunks), overflows[0]


def _open_stream(sd, callback):
    try:
        return sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", callback=callback)
    except sd.PortAudioError as exc:
        device = sd.query_devices(kind="input")
        raise SystemExit(
            f"cannot open input '{device['name']}' at {RATE} Hz "
            f"(native {device['default_samplerate']:.0f} Hz): {exc}"
        ) from exc


def run_session(args: argparse.Namespace) -> None:
    out = Path(args.out)
    people = load_people(INPUTS / "lexicon.local.txt")
    items = load_items(args.kind, people)
    if args.kind == "read":
        _warn_unfilled(people)
    todo = remaining(items, out / "manifest.jsonl", args.speaker, args.mic)
    print(f"{len(todo)} clips left for {args.speaker}/{args.mic} ({args.kind})")
    for n, item in enumerate(todo, 1):
        if not _record_item(out, args, item, n, len(todo)):
            break


def _warn_unfilled(people: dict[str, str]) -> None:
    prompts = (INPUTS / "prompts.jsonl").read_text(encoding="utf-8")
    missing = sorted(set(PERSON_SLOT.findall(prompts)) - set(people))
    if missing:
        print(f"  note: no name for {', '.join(missing)} in lexicon.local.txt; those prompts are skipped")


def _record_item(out: Path, args: argparse.Namespace, item: dict, n: int, total: int) -> bool:
    while True:
        print(f"\n[{n}/{total}] {item['id']}:  {item['show']}")
        start = input("  Enter to start (s skip, q quit) ").strip().lower()
        if start in ("s", "q"):
            return start == "s"
        pcm, overflows = record_until_enter()
        warn = f" WARNING {overflows} input overflows, consider redo." if overflows else ""
        answer = input(f"  {len(pcm) / 2 / RATE:.1f} s.{warn} Enter keep, r redo, q keep+quit ").strip().lower()
        if answer == "r":
            continue
        save_clip(out, args.speaker, args.mic, args.kind, item, pcm)
        return answer != "q"


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speaker", required=True, help="label like spk1, never a real name")
    parser.add_argument("--mic", required=True, help="mic arm, e.g. desk or kitchen")
    parser.add_argument("--out", required=True, help="corpus dir outside the repo")
    parser.add_argument("--kind", choices=("read", "scenario", "nospeech"), default="read")
    run_session(parser.parse_args())


if __name__ == "__main__":
    main()
