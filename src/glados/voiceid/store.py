"""Encrypted voiceprint store (DESIGN-speaker-id.md, Privacy).

One Fernet-encrypted file per enrolled person. The key lives in the OS
keyring (`glados.voiceid` / `store-key`), never on disk beside the prints.
The server reads through `snapshot()`, which reloads when the directory
changes and returns an immutable tuple, so a `forget` from the CLI takes
effect on the next turn without a restart and never mutates a list that an
identification is iterating.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from ..core.secrets import SecretsStore
from .embedder import unit

log = logging.getLogger(__name__)
audit = logging.getLogger("glados.voiceid.audit")

KEY_SCOPE = "voiceid"
KEY_NAME = "store-key"
_SUFFIX = ".vp"
_PERSON_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_REPLACE_RETRIES = 5


class StoreError(Exception):
    pass


@dataclass(frozen=True)
class Voiceprint:
    person: str
    model_id: str
    centroid: np.ndarray = field(repr=False)


def check_person(person: str) -> str:
    if not _PERSON_RE.fullmatch(person):
        raise StoreError(
            f"person id {person!r} must be 1-32 chars of a-z, 0-9, '_' or '-'"
        )
    return person


class VoiceprintStore:
    def __init__(self, root: Path, secrets: SecretsStore) -> None:
        self._root = root
        self._secrets = secrets
        self._signature: tuple | None = None
        self._prints: tuple[Voiceprint, ...] = ()

    def snapshot(self, model_id: str) -> tuple[Voiceprint, ...]:
        signature = self._directory_signature()
        if signature != self._signature:
            self._prints = self._load_all()
            self._signature = signature
            _warn_stale(self._prints, model_id)
        return tuple(p for p in self._prints if p.model_id == model_id)

    def people(self) -> list[tuple[str, str]]:
        return [(p.person, p.model_id) for p in self._load_all()]

    def enroll(
        self,
        person: str,
        model_id: str,
        embeddings: list[np.ndarray],
        *,
        replace: bool = False,
    ) -> Path:
        check_person(person)
        if not embeddings:
            raise StoreError("enrollment needs at least one clip")
        path = self._path(person)
        if path.exists() and not replace:
            raise StoreError(f"{person!r} is already enrolled; pass --replace to overwrite")
        vectors = [unit(e) for e in embeddings]
        record = {
            "person": person,
            "model_id": model_id,
            "embeddings": [v.tolist() for v in vectors],
            "centroid": unit(np.mean(vectors, axis=0)).tolist(),
            "consent": {"enrolled_at": datetime.now(timezone.utc).isoformat()},
        }
        token = self._cipher(create=True).encrypt(json.dumps(record).encode("utf-8"))
        self._atomic_write(path, token)
        audit.info(
            "voiceprint %s person=%s model=%s clips=%d",
            "replaced" if replace else "enrolled", person, model_id, len(vectors),
        )
        return path

    def forget(self, person: str) -> bool:
        path = self._path(check_person(person))
        if not path.exists():
            return False
        _retry_on_lock(path.unlink)
        audit.info("voiceprint forgotten person=%s", person)
        return True

    def _path(self, person: str) -> Path:
        return self._root / f"{person}{_SUFFIX}"

    def _directory_signature(self) -> tuple:
        if not self._root.is_dir():
            return ()
        entries = []
        for p in self._root.glob(f"*{_SUFFIX}"):
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            entries.append((p.name, st.st_mtime_ns, st.st_size))
        return tuple(sorted(entries))

    def _load_all(self) -> tuple[Voiceprint, ...]:
        if not self._root.is_dir():
            return ()
        cipher = self._cipher(create=False)
        if cipher is None:
            log.warning("voiceprint store %s has prints but no keyring key; none loaded", self._root)
            return ()
        loaded = (self._load_one(path, cipher) for path in sorted(self._root.glob(f"*{_SUFFIX}")))
        return tuple(p for p in loaded if p is not None)

    def _load_one(self, path: Path, cipher: Fernet) -> Voiceprint | None:
        try:
            record = json.loads(cipher.decrypt(path.read_bytes()))
            return Voiceprint(
                person=check_person(record["person"]),
                model_id=record["model_id"],
                centroid=unit(record["centroid"]),
            )
        except (OSError, InvalidToken, ValueError, KeyError, TypeError, StoreError) as e:
            log.warning("voiceprint %s skipped: unreadable (%s)", path.name, type(e).__name__)
            return None

    def _cipher(self, *, create: bool) -> Fernet | None:
        key = self._secrets.get(KEY_SCOPE, KEY_NAME)
        if key is None and create:
            self._secrets.set(KEY_SCOPE, KEY_NAME, Fernet.generate_key().decode("ascii"))
            key = self._secrets.get(KEY_SCOPE, KEY_NAME)
        return Fernet(key) if key is not None else None

    def _atomic_write(self, path: Path, data: bytes) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        try:
            _retry_on_lock(lambda: os.replace(tmp, path))
        except OSError:
            tmp.unlink(missing_ok=True)
            raise


def _warn_stale(prints: tuple[Voiceprint, ...], model_id: str) -> None:
    for p in prints:
        if p.model_id != model_id:
            log.warning(
                "voiceprint for %s was made with model %s, not %s; re-enroll needed",
                p.person, p.model_id, model_id,
            )


def _retry_on_lock(operation) -> None:
    """Windows refuses replace/unlink while the server is reading the file."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            operation()
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
