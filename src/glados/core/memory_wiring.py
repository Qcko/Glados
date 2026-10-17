"""Builds per-speaker memory from config (DESIGN-speaker-memory.md).

Kept out of server.py, which only calls `build_speaker_memory` and runs the
writer task in its lifespan.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..brain.llm.fake import FakeLLM
from ..brain.llm.llamacpp import LlamaCppLLM
from ..brain.llm.ollama import OllamaLLM
from .adapters import LLM
from .config import GladosConfig, LLMConfig, MemoryConfig, RoomsConfig
from .memory_writer import MemoryWriter
from .speaker_memory import SpeakerMemoryStore
from .speaker_memory_hooks import SpeakerMemory

if TYPE_CHECKING:
    from .organizer import Organizer

log = logging.getLogger(__name__)


def build_speaker_memory(
    cfg: GladosConfig,
    rooms: RoomsConfig,
    organizer: Callable[[], "Organizer"],
) -> tuple[SpeakerMemory, MemoryWriter, LLM] | None:
    """None when memory is off. `organizer` is late-bound because the
    organizer is built with the result of this call."""
    if not cfg.memory.enabled:
        return None
    store = SpeakerMemoryStore(cfg.memory.db_path, max_facts=cfg.memory.max_facts)
    writer_llm = build_writer_llm(cfg.llm, cfg.memory)
    writer = MemoryWriter(
        llm=writer_llm,
        store=store,
        is_busy=lambda: organizer().any_turn_busy(),
        is_warm=lambda: organizer().llm_warmed,
        rewarm=lambda: organizer().rewarm_prefix(),
        idle_s=cfg.memory.idle_s,
    )
    memory = SpeakerMemory(
        store=store,
        writer=writer,
        room_enabled=lambda room_id: _room_opted_in(rooms, room_id),
        inject_chars=cfg.memory.inject_chars,
    )
    log.info(
        "speaker memory on: rooms=%s inject_chars=%d",
        [r.room_id for r in rooms.rooms if r.memory], cfg.memory.inject_chars,
    )
    return memory, writer, writer_llm


def _room_opted_in(rooms: RoomsConfig, room_id: str) -> bool:
    policy = rooms.policy_for(room_id)
    return policy is not None and policy.memory


def build_writer_llm(cfg: LLMConfig, mem: MemoryConfig) -> LLM:
    """The writer's own adapter: the primary's host and EVERY load-relevant
    option (num_ctx, keep_alive), so a writer call never reloads the resident
    model. Only the generation cap differs, which Ollama applies per request."""
    model = mem.writer_model or cfg.model
    if model != cfg.model:
        log.warning(
            "memory writer_model %s differs from [llm].model %s: a second model "
            "will load beside the primary",
            model, cfg.model,
        )
    if cfg.backend == "ollama":
        return OllamaLLM(
            host=cfg.host,
            model=model,
            temperature=0.0,
            timeout=cfg.timeout,
            keep_alive=cfg.keep_alive,
            text_tool_format=None,
            num_ctx=cfg.num_ctx,
            num_predict=mem.writer_num_predict,
            repeat_penalty=cfg.repeat_penalty,
            think=cfg.think,
        )
    if cfg.backend == "llamacpp":
        return LlamaCppLLM(
            host=cfg.llamacpp_host,
            model=model,
            temperature=0.0,
            timeout=cfg.timeout,
            max_tokens=mem.writer_num_predict,
            repeat_penalty=cfg.repeat_penalty,
            num_ctx=cfg.num_ctx,
            api_key=(
                os.environ.get(cfg.llamacpp_api_key_env)
                if cfg.llamacpp_api_key_env
                else None
            ),
        )
    if cfg.backend == "fake":
        return FakeLLM()
    raise ValueError(f"[llm] backend = {cfg.backend!r} has no memory writer builder")
