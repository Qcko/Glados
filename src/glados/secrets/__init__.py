"""CLI for populating the OS keyring used by the server.

Importable: `from glados.secrets import run` exposes the CLI entry point
used by `python -m glados.secrets`.

Run with:
    python -m glados.secrets set client-tokens desk-ui
    python -m glados.secrets get client-tokens desk-ui
    python -m glados.secrets delete client-tokens desk-ui

(There's no `list` subcommand -- the `keyring` package doesn't expose a
portable enumeration API; rely on `glados.toml`'s `[auth] clients` list
or your OS credential manager UI to see what's stored.)

`set` reads the value from stdin via getpass -- no shell history, no echo.
"""

from .__main__ import run

__all__ = ["run"]
