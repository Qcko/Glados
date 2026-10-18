"""Open-set speaker identification (DESIGN-speaker-id.md, Decisions 2-3).

`identify` never raises and never blocks a turn for longer than its
timeout: every failure is a SpeakerMatch with person None and a reason.
Embedding runs on a dedicated single-worker executor, so a slow model can
neither starve the executor STT uses nor pile up threads -- while one embed
is still running, the next utterance is answered `busy` at once.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.speaker import SpeakerMatch
from .embedder import SAMPLE_RATE, SpeakerEmbedder, unit
from .store import Voiceprint, VoiceprintStore

log = logging.getLogger(__name__)


class SpeakerIdentifier:
    def __init__(
        self,
        embedder: SpeakerEmbedder,
        store: VoiceprintStore,
        *,
        accept_threshold: float,
        margin: float,
        min_speech_s: float,
        timeout_s: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._accept = accept_threshold
        self._margin = margin
        self._min_samples = int(min_speech_s * SAMPLE_RATE)
        self._timeout_s = timeout_s
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voiceid")
        self._busy = False

    async def identify(self, pcm: bytes) -> SpeakerMatch:
        try:
            prints = self._store.snapshot(self._embedder.model_id)
        except Exception:
            log.exception("voiceprint store unreadable; identifying nobody")
            return SpeakerMatch.none("error")
        if not prints:
            return SpeakerMatch.none("no_prints")
        if len(pcm) // 2 < self._min_samples:
            return SpeakerMatch.none("too_short")
        if self._busy:
            return SpeakerMatch.none("busy")
        return await self._embed_and_decide(pcm, prints)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _embed_and_decide(
        self, pcm: bytes, prints: tuple[Voiceprint, ...]
    ) -> SpeakerMatch:
        try:
            job = self._executor.submit(self._embedder.embed, pcm)
        except RuntimeError:
            log.warning("speaker identifier is closed; turn goes unidentified")
            return SpeakerMatch.none("error")
        self._busy = True
        job.add_done_callback(self._mark_idle)
        try:
            embedding = await asyncio.wait_for(asyncio.wrap_future(job), self._timeout_s)
        except asyncio.TimeoutError:
            log.warning("speaker embed exceeded %.0f ms; turn goes unidentified", self._timeout_s * 1000)
            return SpeakerMatch.none("timeout")
        except Exception as e:
            log.error("speaker embed failed (%s); turn goes unidentified", type(e).__name__)
            return SpeakerMatch.none("error")
        return decide(embedding, prints, accept_threshold=self._accept, margin=self._margin)

    def _mark_idle(self, _job) -> None:
        self._busy = False


def decide(
    embedding: np.ndarray,
    prints: tuple[Voiceprint, ...],
    *,
    accept_threshold: float,
    margin: float,
) -> SpeakerMatch:
    """Best enrolled person if the match is strong AND clearly ahead of the
    runner-up; with one enrolled person the runner-up scores 0."""
    probe = unit(embedding)
    ranked = sorted(
        ((float(np.dot(probe, p.centroid)), p.person) for p in prints), reverse=True
    )
    best_score, best_person = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < accept_threshold:
        return SpeakerMatch.none("low_score", best_score)
    if best_score - runner_up < margin:
        return SpeakerMatch.none("low_margin", best_score)
    return SpeakerMatch(person=best_person, score=best_score, reason="accepted")
