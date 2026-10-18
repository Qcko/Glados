"""Durable per-speaker facts (DESIGN-speaker-memory.md).

One SQLite file in WAL mode, two connections: a READ connection used on the
event loop for the turn-start lookup, and a WRITE connection used only from
worker threads (`asyncio.to_thread`) behind a lock. WAL lets the read run
while a write is in progress, so a turn never waits on the writer's fsync.

Facts are keyed `(room_id, speaker_id)`, where `speaker_id` is always a
voice-verified person (`SpeakerIdentity.person`), never the session bucket
`session.speaker_id`, which falls back to the client's default user.

A per-key `generation` is bumped by every write, including the operator CLI
running in another process, and `purges` by every delete. The writer's apply
is refused if the generation moved since its job read the facts, if a purge
happened since the job's first turn was noted, or if the job's session was
forgotten -- so nothing the operator removed comes back from notes the writer
was still holding. The READ side uses `PRAGMA data_version` to notice writes made
by any other connection and reload its cache.

This table is NOT the section 8 "SQLite session log" (unbuilt); the names
`speaker_facts` / `speaker_generation` are reserved so that log does not fork
this schema.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

from .memory_facts import Fact

log = logging.getLogger(__name__)

Key = tuple[str, str]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    text TEXT NOT NULL,
    source_session TEXT NOT NULL,
    source_utterance TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS speaker_facts_key ON speaker_facts(room_id, speaker_id);
CREATE TABLE IF NOT EXISTS speaker_generation (
    room_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    purges INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (room_id, speaker_id)
);
CREATE TABLE IF NOT EXISTS speaker_forgotten_sessions (
    session_id TEXT PRIMARY KEY
);
"""


class StaleGeneration(Exception):
    """The speaker's facts changed since the job read them."""


class SpeakerMemoryStore:
    def __init__(self, path: Path, *, max_facts: int = 30) -> None:
        self._path = Path(path)
        self._max_facts = max_facts
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write = self._connect(check_same_thread=False)
        self._write.executescript(_SCHEMA)
        _restrict_to_owner(self._path)
        self._read = self._connect(check_same_thread=False)
        self._cache: dict[Key, list[Fact]] = {}
        self._seen_version: int | None = None

    def _connect(self, *, check_same_thread: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._path, check_same_thread=check_same_thread, isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=2000")
        return conn

    def close(self) -> None:
        with self._read_lock:
            self._read.close()
        with self._write_lock:
            self._write.close()

    def facts(self, room_id: str, speaker_id: str) -> list[Fact]:
        """Newest first. Served from cache; a write by any connection since the
        last call (the writer thread, the CLI) reloads it first."""
        key = (room_id, speaker_id)
        with self._read_lock:
            self._refresh_if_changed()
            if key not in self._cache:
                self._cache[key] = self._load(key)
            return list(self._cache[key])

    def generation(self, room_id: str, speaker_id: str) -> int:
        """Read this BEFORE the facts a write is based on: a write landing in
        between then makes the generation stale and the write is refused."""
        return self.stamp(room_id, speaker_id)[0]

    def purges(self, room_id: str, speaker_id: str) -> int:
        return self.stamp(room_id, speaker_id)[1]

    def stamp(self, room_id: str, speaker_id: str) -> tuple[int, int]:
        with self._read_lock:
            return _stamp(self._read, (room_id, speaker_id))

    def _refresh_if_changed(self) -> None:
        version = self._read.execute("PRAGMA data_version").fetchone()[0]
        if version != self._seen_version:
            self._seen_version = version
            self._cache.clear()

    def _load(self, key: Key) -> list[Fact]:
        rows = self._read.execute(
            "SELECT id, text FROM speaker_facts WHERE room_id=? AND speaker_id=? "
            "ORDER BY last_seen_at DESC, id DESC",
            key,
        ).fetchall()
        return [Fact(id=r[0], text=r[1]) for r in rows]

    async def apply(
        self,
        key: Key,
        *,
        adds: list[str],
        drop_ids: list[int],
        expected_generation: int,
        expected_purges: int = 0,
        source_session: str,
        source_utterance: str,
    ) -> None:
        await asyncio.to_thread(
            self._apply_sync, key, adds, drop_ids,
            (expected_generation, expected_purges), source_session, source_utterance,
        )

    def _apply_sync(
        self, key: Key, adds: list[str], drop_ids: list[int],
        expected: tuple[int, int], source_session: str, source_utterance: str,
    ) -> None:
        with self._write_lock:
            conn = self._write
            conn.execute("BEGIN IMMEDIATE")
            try:
                if _stamp(conn, key) != expected or _forgotten(conn, source_session):
                    raise StaleGeneration(key)
                _delete_ids(conn, key, drop_ids)
                now = time.time()
                for text in adds:
                    conn.execute(
                        "INSERT INTO speaker_facts (room_id, speaker_id, text, "
                        "source_session, source_utterance, created_at, last_seen_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (*key, text, source_session, source_utterance, now, now),
                    )
                self._evict_over_cap(conn, key)
                _bump_generation(conn, key)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _evict_over_cap(self, conn: sqlite3.Connection, key: Key) -> None:
        conn.execute(
            "DELETE FROM speaker_facts WHERE room_id=? AND speaker_id=? AND id NOT IN ("
            "SELECT id FROM speaker_facts WHERE room_id=? AND speaker_id=? "
            "ORDER BY last_seen_at DESC, id DESC LIMIT ?)",
            (*key, *key, self._max_facts),
        )

    def list_all(self) -> list[tuple[int, str, str, str, str]]:
        with self._read_lock:
            return self._read.execute(
                "SELECT id, room_id, speaker_id, text, source_session FROM speaker_facts "
                "ORDER BY room_id, speaker_id, last_seen_at DESC"
            ).fetchall()

    def forget_fact(self, fact_id: int) -> int:
        return self._delete_where("id=?", (fact_id,))

    def forget_session(self, session_id: str) -> int:
        return self._delete_where(
            "source_session=?", (session_id,), forget_session=session_id
        )

    def wipe(self, room_id: str, speaker_id: str) -> int:
        return self._delete_where(
            "room_id=? AND speaker_id=?", (room_id, speaker_id),
            purge_key=(room_id, speaker_id),
        )

    def _delete_where(
        self, where: str, args: tuple, *,
        forget_session: str | None = None, purge_key: Key | None = None,
    ) -> int:
        with self._write_lock:
            conn = self._write
            conn.execute("BEGIN IMMEDIATE")
            try:
                keys = conn.execute(
                    f"SELECT DISTINCT room_id, speaker_id FROM speaker_facts WHERE {where}",
                    args,
                ).fetchall()
                count = conn.execute(f"DELETE FROM speaker_facts WHERE {where}", args).rowcount
                for key in {tuple(k) for k in keys} | ({purge_key} if purge_key else set()):
                    _bump_generation(conn, key, purge=True)
                if forget_session is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO speaker_forgotten_sessions VALUES (?)",
                        (forget_session,),
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return count


def _stamp(conn: sqlite3.Connection, key: Key) -> tuple[int, int]:
    row = conn.execute(
        "SELECT generation, purges FROM speaker_generation WHERE room_id=? AND speaker_id=?",
        key,
    ).fetchone()
    return (row[0], row[1]) if row else (0, 0)


def _forgotten(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM speaker_forgotten_sessions WHERE session_id=?", (session_id,)
    ).fetchone() is not None


def _bump_generation(conn: sqlite3.Connection, key: Key, *, purge: bool = False) -> None:
    conn.execute(
        "INSERT INTO speaker_generation (room_id, speaker_id, generation, purges) "
        "VALUES (?, ?, 1, ?) ON CONFLICT(room_id, speaker_id) DO UPDATE SET "
        "generation = generation + 1, purges = purges + excluded.purges",
        (*key, int(purge)),
    )


def _delete_ids(conn: sqlite3.Connection, key: Key, ids: list[int]) -> None:
    for fact_id in ids:
        conn.execute(
            "DELETE FROM speaker_facts WHERE id=? AND room_id=? AND speaker_id=?",
            (fact_id, *key),
        )


def _restrict_to_owner(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        log.warning("speaker memory: could not restrict file mode on the db")
