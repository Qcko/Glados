from __future__ import annotations

import argparse
import getpass
import sys
from typing import Callable, Sequence

from ..core.secrets import KeyringSecrets, SecretsStore


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
    return 2


if __name__ == "__main__":
    raise SystemExit(run())
