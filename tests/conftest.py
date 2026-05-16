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
