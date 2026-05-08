"""Config loader for `glados.toml` and `rooms.toml`.

Schema is intentionally tiny in v0 — fields will accrete as adapters land.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .protocols import Role


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    traces_dir: Path = Path("traces")


class AuthConfig(BaseModel):
    tokens: dict[str, str] = {}


class LLMConfig(BaseModel):
    backend: Literal["fake", "ollama"] = "fake"
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: float = 60.0


class GladosConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()
    llm: LLMConfig = LLMConfig()


class ClientBinding(BaseModel):
    client_id: str
    room_id: str
    role: Role
    default_user: str = "default"


class RoomsConfig(BaseModel):
    clients: list[ClientBinding] = []

    def find(self, client_id: str) -> ClientBinding | None:
        return next((c for c in self.clients if c.client_id == client_id), None)


def load_glados_config(path: Path) -> GladosConfig:
    return _apply_env_overrides(GladosConfig(**_read_toml(path)))


def _apply_env_overrides(cfg: GladosConfig) -> GladosConfig:
    updates: dict = {}
    backend = os.environ.get("GLADOS_LLM_BACKEND")
    if backend in ("fake", "ollama"):
        updates["backend"] = backend
    if (model := os.environ.get("GLADOS_LLM_MODEL")) is not None:
        updates["model"] = model
    if (host := os.environ.get("GLADOS_LLM_HOST")) is not None:
        updates["host"] = host
    if updates:
        cfg = cfg.model_copy(update={"llm": cfg.llm.model_copy(update=updates)})
    return cfg


def load_rooms_config(path: Path) -> RoomsConfig:
    return RoomsConfig(**_read_toml(path))


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)
