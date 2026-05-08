"""Config loader for `glados.toml` and `rooms.toml`.

Schema is intentionally tiny in v0 — fields will accrete as adapters land.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

from .protocols import Role


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    traces_dir: Path = Path("traces")


class AuthConfig(BaseModel):
    tokens: dict[str, str] = {}


class GladosConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    auth: AuthConfig = AuthConfig()


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
    return GladosConfig(**_read_toml(path))


def load_rooms_config(path: Path) -> RoomsConfig:
    return RoomsConfig(**_read_toml(path))


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)
