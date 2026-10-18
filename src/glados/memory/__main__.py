"""Operator CLI for per-speaker memory (DESIGN-speaker-memory.md).

Shell access on the server is the only way in, by design: this is how a fact
that should not be there gets removed until a voice "forget that" exists.
Safe to run while GLaDOS is up -- every delete bumps the speaker's generation,
so an in-flight batch chunk cannot re-add what was just removed, and the
server's cache reloads on its next turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import aclosing
from pathlib import Path
from typing import Sequence

from ..core.adapters import LLMText
from ..core.config import load_glados_config
from ..core.memory_facts import DayBudget, StagedNote, judge_writer_output, writer_messages
from ..core.memory_wiring import batch_policy, build_writer_llm
from ..core.speaker_memory import SpeakerMemoryStore
from . import bakeoff, scoring


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m glados.memory",
        description="Inspect and remove per-speaker memory.",
    )
    p.add_argument("--db", type=Path, help="Database path (default: [memory].db_path).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Print every stored fact.")
    p_forget = sub.add_parser("forget", help="Delete one fact by id, and any staged note it cites.")
    p_forget.add_argument("fact_id", type=int)
    p_session = sub.add_parser(
        "forget-session", help="Delete every fact one session backed, and its staged notes."
    )
    p_session.add_argument("session_id")
    p_wipe = sub.add_parser("wipe", help="Delete every fact and staged note for a room's speaker.")
    p_wipe.add_argument("room_id")
    p_wipe.add_argument("speaker_id")
    p_replay = sub.add_parser(
        "replay",
        help="Run a fixed transcript through a writer model and print what it "
        "would store (bake-off harness; touches no database).",
    )
    p_replay.add_argument("transcript", type=Path, help='JSON list of {"user": ..., "reply": ...}.')
    p_replay.add_argument("--model", default="", help="Writer model (default: [memory].writer_model).")
    p_bake = sub.add_parser(
        "bakeoff",
        help="Score a writer model on the labelled corpus through the real "
        "nightly fold (local ollama only, GLaDOS stopped).",
    )
    p_bake.add_argument("--model", default="", help="Writer model (default: [memory].writer_model or [llm].model).")
    p_bake.add_argument("--cases", type=Path, action="append",
                        help="Corpus file(s) (default: benchmarks/memory/cases.jsonl).")
    p_bake.add_argument("--out", type=Path, help="Report dir (default: $GLADOS_LAB_DIR/glados/memory_bakeoff).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "replay":
        return asyncio.run(_replay(args.transcript, args.model))
    if args.cmd == "bakeoff":
        return asyncio.run(_bakeoff(args))
    store = SpeakerMemoryStore(args.db or _configured_db())
    try:
        return _run(store, args)
    finally:
        store.close()


def _configured_db() -> Path:
    return _config().memory.db_path


def _config():
    config_dir = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))
    return load_glados_config(config_dir / "glados.toml")


async def _bakeoff(args: argparse.Namespace) -> int:
    cfg = _config()
    mem = cfg.memory.model_copy(update={"writer_model": args.model}) if args.model else cfg.memory
    model = mem.writer_model or cfg.llm.model
    try:
        bakeoff.check_local_ollama(cfg)
        bakeoff.check_glados_stopped(cfg)
        digest = bakeoff.model_digest(cfg.llm.host, model)
    except bakeoff.BakeoffRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    paths = args.cases or [bakeoff.default_cases()]
    texts = [p.read_text(encoding="utf-8") for p in paths]
    cases = [c for text in texts for c in scoring.load_cases(text)]
    llm = build_writer_llm(cfg.llm, mem)
    try:
        scores = await bakeoff.run_cases(cases, llm, batch_policy(cfg))
    finally:
        aclose = getattr(llm, "aclose", None)
        if aclose is not None:
            await aclose()
        bakeoff.unload_quietly(cfg.llm.host, model)
    data = bakeoff.report(scores, {
        "model": model, "digest": digest, "corpus_sha256": bakeoff.corpus_sha256(texts),
        "settings": {"num_ctx": cfg.llm.num_ctx, "num_predict": mem.writer_num_predict,
                     "think": cfg.llm.think, "daily_adds": mem.daily_adds,
                     "max_notes_per_batch": mem.max_notes_per_batch},
    })
    json_path, md_path = bakeoff.write_report(data, args.out or bakeoff.default_out_dir())
    test_summary = scoring.summarise([s for s in scores if s.split == "test"])
    print(bakeoff.headline(test_summary))
    print(f"report: {md_path}")
    if any(s.failed_chunks for s in scores):
        print("INVALID: writer calls failed; see the memory log for why", file=sys.stderr)
        return 3
    return 0 if test_summary.gate == "pass" else 4 if test_summary.gate == "pass" else 4


async def _replay(transcript: Path, model: str) -> int:
    cfg = _config()
    mem = cfg.memory.model_copy(update={"writer_model": model}) if model else cfg.memory
    llm = build_writer_llm(cfg.llm, mem)
    notes = [
        StagedNote(id=i, user_text=t["user"], reply=t.get("reply"))
        for i, t in enumerate(json.loads(transcript.read_text(encoding="utf-8")), start=1)
    ]
    parts: list[str] = []
    try:
        async with aclosing(llm.chat(writer_messages([], notes), [])) as stream:
            async for event in stream:
                if isinstance(event, LLMText):
                    parts.append(event.text)
    finally:
        aclose = getattr(llm, "aclose", None)
        if aclose is not None:
            await aclose()
    raw = "".join(parts)
    verdict = judge_writer_output(raw, [], notes, DayBudget(adds=mem.daily_adds, drops=0, untrusted_adds=0))
    adds = [{"fact": a.text, "notes": list(a.note_ids)} for a in verdict.adds]
    print(json.dumps({"raw": raw, "adds": adds, "rejected": verdict.rejected}, indent=2))
    return 0


def _run(store: SpeakerMemoryStore, args: argparse.Namespace) -> int:
    if args.cmd == "list":
        for fact_id, room, speaker, text, session in store.list_all():
            print(f"{fact_id}\t{room}\t{speaker}\t{session}\t{text}")
        return 0
    if args.cmd == "forget":
        removed = store.forget_fact(args.fact_id)
    elif args.cmd == "forget-session":
        removed = store.forget_session(args.session_id)
    else:
        removed = store.wipe(args.room_id, args.speaker_id)
    print(f"removed {removed.facts} facts, {removed.notes} staged notes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
