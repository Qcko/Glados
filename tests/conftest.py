"""Shared test setup.

Default the heavy adapters to "fake" so importing `glados.core.server`
in any test fixture doesn't pull in Ollama / silero-vad / faster-whisper.
Integration tests that want the real thing override these in their own
fixture *before* `import glados.core.server`.
"""

from __future__ import annotations

import os

os.environ.setdefault("GLADOS_LLM_BACKEND", "fake")
os.environ.setdefault("GLADOS_VAD_BACKEND", "fake")
os.environ.setdefault("GLADOS_STT_BACKEND", "fake")
os.environ.setdefault("GLADOS_TTS_BACKEND", "fake")
# Pin the v2.6 router OFF for the default app, independent of whatever the
# shipped configs/glados.toml sets. The end-to-end wire-sequence tests assert
# exact frame order and must not gain a `route_notice` just because routing is
# enabled by default; routing itself is covered in test_router.py.
os.environ.setdefault("GLADOS_ROUTER_ENABLED", "false")

import pytest

from glados.core.secrets import InMemorySecrets


# Default dev tokens matching configs/glados.toml's [auth] clients list.
# Real deployments populate the OS keyring via `python -m glados.secrets`.
_DEV_TOKENS = {
    ("client-tokens", "desk-ui"): "dev-token-desk",
    ("client-tokens", "desk2-ui"): "dev-token-desk2",
}


@pytest.fixture(autouse=True, scope="session")
def _patch_keyring_for_tests():
    """Replace KeyringSecrets with a dev-seeded InMemorySecrets at the
    class symbol, so every `build_app()` -- module-level default or a
    fresh one a test constructs -- gets a safe store by construction.
    Tests never touch the OS keyring."""
    import glados.core.server as srv

    real_cls = srv.KeyringSecrets
    srv.KeyringSecrets = lambda: InMemorySecrets(_DEV_TOKENS)  # type: ignore[assignment]
    try:
        # Re-seed the already-built module-level `app.state.secrets` since
        # `build_app()` ran at import time before this patch took effect.
        srv.app.state.secrets = InMemorySecrets(_DEV_TOKENS)
        yield
    finally:
        srv.KeyringSecrets = real_cls
