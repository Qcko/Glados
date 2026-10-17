from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

from ..core.config import load_glados_config
from ..core.secrets import KeyringSecrets, SecretsStore
from .ingest import ingest_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m glados.secrets",
        description="Manage GLaDOS secrets in the OS keyring.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Store a value (prompted from stdin).")
    p_set.add_argument("scope")
    p_set.add_argument("name")

    p_get = sub.add_parser("get", help="Print a value to stdout (handle with care).")
    p_get.add_argument("scope")
    p_get.add_argument("name")

    p_del = sub.add_parser("delete", help="Delete a value.")
    p_del.add_argument("scope")
    p_del.add_argument("name")

    p_ingest = sub.add_parser(
        "ingest",
        help="Store every allowlisted <client>.token (and the reserved "
        "admin.observe-token.token) in a directory, then delete it.",
    )
    p_ingest.add_argument("inbox", type=Path)
    p_ingest.add_argument("--summary", type=Path, help="Write outcomes (no values) as JSON here.")

    return p


def run(
    argv: Sequence[str] | None = None,
    *,
    store: SecretsStore | None = None,
    prompt: Callable[[str], str] = getpass.getpass,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    args = _build_parser().parse_args(argv)
    store = store or KeyringSecrets()

    if args.cmd == "set":
        value = prompt(f"Value for {args.scope}/{args.name}: ")
        if not value:
            print("aborted: empty value", file=err)
            return 2
        store.set(args.scope, args.name, value)
        print(f"stored {args.scope}/{args.name}", file=out)
        return 0
    if args.cmd == "get":
        value = store.get(args.scope, args.name)
        if value is None:
            print(f"not found: {args.scope}/{args.name}", file=err)
            return 1
        print(value, file=out)
        return 0
    if args.cmd == "delete":
        ok = store.delete(args.scope, args.name)
        if not ok:
            print(f"not found: {args.scope}/{args.name}", file=err)
            return 1
        print(f"deleted {args.scope}/{args.name}", file=out)
        return 0
    if args.cmd == "ingest":
        return _ingest(args.inbox, args.summary, store, out)
    return 2


def _ingest(inbox: Path, summary: Path | None, store: SecretsStore, out) -> int:
    outcomes = ingest_dir(inbox, _allowed_clients(), store)
    for outcome in outcomes:
        print(f"{outcome.result} {outcome.name} {outcome.reason}".rstrip(), file=out)
    if summary is not None:
        summary.write_text(json.dumps([o.as_dict() for o in outcomes], indent=2), encoding="utf-8")
    return 1 if any(o.result == "kept" or o.reason for o in outcomes) else 0


def _allowed_clients() -> list[str]:
    config_dir = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))
    return load_glados_config(config_dir / "glados.toml").auth.clients


if __name__ == "__main__":
    raise SystemExit(run())
