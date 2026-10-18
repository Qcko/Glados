"""Voice identification (ARCH v5, DESIGN-speaker-id.md).

Public surface: `build_identifier` for the server, `SpeakerIdentifier` and
`VoiceprintStore` for tests and the enrollment CLI (`python -m glados.voiceid`).
"""

from __future__ import annotations

import logging

from ..core.config import SpeakerIdConfig
from ..core.secrets import SecretsStore
from .embedder import FakeEmbedder, SpeakerEmbedder
from .identifier import SpeakerIdentifier
from .store import VoiceprintStore

log = logging.getLogger(__name__)

__all__ = ["SpeakerIdentifier", "VoiceprintStore", "build_embedder", "build_identifier"]


def build_embedder(cfg: SpeakerIdConfig) -> SpeakerEmbedder | None:
    if cfg.backend == "fake":
        return FakeEmbedder()
    return None


def build_identifier(cfg: SpeakerIdConfig, secrets: SecretsStore) -> SpeakerIdentifier | None:
    """None means every voice turn is `fallback`. A backend that fails to
    load degrades to that rather than failing boot."""
    try:
        embedder = build_embedder(cfg)
    except Exception:
        log.exception("speaker-id backend %r failed to load; voices go unidentified", cfg.backend)
        return None
    if embedder is None:
        log.info("speaker-id off; every voice turn is unidentified")
        return None
    log.info("speaker-id on: backend=%s model=%s", cfg.backend, embedder.model_id)
    return SpeakerIdentifier(
        embedder,
        VoiceprintStore(cfg.store_dir, secrets),
        accept_threshold=cfg.accept_threshold,
        margin=cfg.margin,
        min_speech_s=cfg.min_speech_s,
        timeout_s=cfg.timeout_ms / 1000,
    )
