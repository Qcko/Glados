"""Enrollment CLI: `python -m glados.voiceid enroll|list|forget`.

Server-side only, by design: there is no MCP tool or voice command that
enrolls anyone (DESIGN-speaker-id.md, Privacy). Clips are 16 kHz mono
16-bit WAV, ideally recorded through the room mic the person will use; the
audio is embedded and discarded, never stored.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import wave
from pathlib import Path
from typing import Callable, Sequence

from ..core.config import load_glados_config, load_rooms_config
from ..core.secrets import KeyringSecrets, SecretsStore
from .embedder import SAMPLE_RATE, SpeakerEmbedder
from .store import StoreError, VoiceprintStore
from . import build_embedder

_MIN_CLIPS_ADVISED = 3


def main(
    argv: Sequence[str] | None = None,
    *,
    secrets: SecretsStore | None = None,
    embedder_factory: Callable[..., SpeakerEmbedder | None] = build_embedder,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    cfg_dir = args.config_dir or Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))
    cfg = load_glados_config(cfg_dir / "glados.toml").speaker_id
    store = VoiceprintStore(cfg.store_dir, secrets or KeyringSecrets())
    try:
        if args.cmd == "list":
            return _list(store)
        if args.cmd == "forget":
            return _forget(store, args.person)
        embedder = embedder_factory(cfg)
        if embedder is None:
            print("speaker_id.backend is 'off' in glados.toml; set a backend first", file=sys.stderr)
            return 2
        return _enroll(store, embedder, args, _known_users(cfg_dir))
    except StoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m glados.voiceid", description="Manage voiceprints.")
    p.add_argument("--config-dir", type=Path, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    enroll = sub.add_parser("enroll", help="Enroll a person from WAV clips.")
    enroll.add_argument("person")
    enroll.add_argument("clips", nargs="+", type=Path)
    enroll.add_argument("--replace", action="store_true", help="Overwrite an existing voiceprint.")
    enroll.add_argument("--new", action="store_true", help="Allow a person no room lists as default_user.")
    sub.add_parser("list", help="List enrolled people (no voice data).")
    forget = sub.add_parser("forget", help="Delete a person's voiceprint.")
    forget.add_argument("person")
    return p


def _enroll(store: VoiceprintStore, embedder: SpeakerEmbedder, args, known: set[str]) -> int:
    if args.person not in known and not args.new:
        print(
            f"error: {args.person!r} is not a default_user in rooms.toml "
            f"(known: {', '.join(sorted(known))}); pass --new if that is intended",
            file=sys.stderr,
        )
        return 1
    embeddings = [embedder.embed(_read_clip(path)) for path in args.clips]
    if len(embeddings) < _MIN_CLIPS_ADVISED:
        print(f"warning: {len(embeddings)} clip(s); {_MIN_CLIPS_ADVISED}+ give a steadier print")
    store.enroll(args.person, embedder.model_id, embeddings, replace=args.replace)
    print(f"enrolled {args.person} ({len(embeddings)} clips, model {embedder.model_id})")
    return 0


def _list(store: VoiceprintStore) -> int:
    for person, model_id in store.people():
        print(f"{person}\t{model_id}")
    return 0


def _forget(store: VoiceprintStore, person: str) -> int:
    if store.forget(person):
        print(f"forgot {person}")
        return 0
    print(f"{person} is not enrolled", file=sys.stderr)
    return 1


def _known_users(cfg_dir: Path) -> set[str]:
    return {c.default_user for c in load_rooms_config(cfg_dir / "rooms.toml").clients}


def _read_clip(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise StoreError(f"{path.name}: need {SAMPLE_RATE} Hz mono 16-bit WAV")
        return w.readframes(w.getnframes())


if __name__ == "__main__":
    sys.exit(main())
