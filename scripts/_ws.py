"""Shared websocket-client bits for the dev scripts.

Deliberately NOT in `glados.core`: the only thing here disables TLS
verification, and a helper that does that should not be importable from the
shipped package, where it would eventually be reused by something that
shouldn't.
"""

from __future__ import annotations

import ssl
from urllib.parse import urlparse

# The dev certificate is self-signed, so the default context rejects it and the
# connection never opens. That is a fine trade against a server on this machine
# and a terrible one against anything else.
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def ssl_context(url: str) -> ssl.SSLContext | None:
    """A context for `wss://`, or None for `ws://` (where websockets rejects a
    context outright).

    Verification is skipped only for loopback. The URL is overridable by env
    var in both scripts, so "it's just the dev cert" stops being true the moment
    someone points one at another host -- and a script that silently accepts any
    certificate is how that becomes a habit."""
    if not url.startswith("wss://"):
        return None
    ctx = ssl.create_default_context()
    if (urlparse(url).hostname or "") in _LOOPBACK:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx
