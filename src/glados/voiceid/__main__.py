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
from . import ONNX_BACKENDS, bakeoff, build_embedder, compare

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
        if args.cmd == "bakeoff":
            return _bakeoff(args, cfg)
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
    bake = sub.add_parser(
        "bakeoff", help="Pick accept_threshold from household clips (<dir>/<person>/, <dir>/impostor/).",
    )
    bake.add_argument("clip_dir", type=Path)
    bake.add_argument(
        "--backend", type=_backends, default=("wespeaker",),
        help="One backend, or 'wespeaker,ecapa' for the side-by-side comparison.",
    )
    bake.add_argument(
        "--model", type=Path, action="append",
        help="ONNX file per backend, in --backend order (one backend: default [speaker_id].model_path).",
    )
    bake.add_argument("--out", type=Path, help="Report dir (default: $GLADOS_LAB_DIR/glados/voiceid_bakeoff).")
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


def _backends(text: str) -> tuple[str, ...]:
    names = tuple(n.strip() for n in text.split(",") if n.strip())
    unknown = [n for n in names if n not in ONNX_BACKENDS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown backend(s) {unknown}; choose from {sorted(ONNX_BACKENDS)}")
    if len(names) > 1 and sorted(names) != sorted(compare.BACKENDS):
        raise argparse.ArgumentTypeError(f"side-by-side compares exactly {','.join(compare.BACKENDS)}")
    return names


def _bakeoff(args, cfg) -> int:
    models = args.model or [cfg.model_path]
    if len(models) != len(args.backend):
        print(f"error: {len(args.backend)} backend(s) need {len(args.backend)} --model", file=sys.stderr)
        return 2
    try:
        embedders = _load_embedders(args.backend, models, cfg.threads)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        if len(embedders) > 1:
            return _report_comparison(args, compare.run(args.clip_dir, embedders, cfg.margin))
        return _report_single(args, bakeoff.run(args.clip_dir, embedders[args.backend[0]], cfg.margin))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _load_embedders(backends, models, threads: int) -> dict[str, SpeakerEmbedder]:
    embedders = {}
    for name, model in zip(backends, models):
        try:
            embedders[name] = ONNX_BACKENDS[name](model, threads=threads)
        except Exception as e:
            raise RuntimeError(f"backend {name} cannot load {Path(model).name}: {e}") from e
    return embedders


def _report_single(args, data: dict) -> int:
    path = bakeoff.write_report(data, args.out or bakeoff.default_out_dir())
    print(bakeoff.headline(data))
    print(f"report: {path}")
    if not data["pick"]:
        return 4
    return 0 if data["enough_impostors"] else 5


def _report_comparison(args, data: dict) -> int:
    path = bakeoff.write_report(data, args.out or bakeoff.default_out_dir())
    print(compare.headline(data))
    print(f"report: {path}")
    if data["verdict"]["winner"] is None:
        return 4
    return 5 if data["verdict"]["provisional"] else 0


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
