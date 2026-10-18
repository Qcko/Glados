"""Durable per-speaker facts and the day's staged notes (DESIGN-speaker-memory.md).

One SQLite file in WAL mode, two connections: a READ connection used on the
event loop for the turn-start lookup, and a WRITE connection used only from
worker threads (`asyncio.to_thread`) behind a lock. WAL lets the read run
while a write is in progress, so a turn never waits on the writer's fsync.

Facts are keyed `(room_id, speaker_id)`, where `speaker_id` is always a
voice-verified person (`SpeakerIdentity.person`), never the session bucket
`session.speaker_id`, which falls back to the client's default user.

Staged notes are the day's verbatim speech, waiting for the nightly batch.
Their text is Fernet ciphertext under a keyring key of its own; with no key
nothing is staged. A note is deleted in the same transaction that writes what
it produced, so a cancel or a crash can neither lose nor double-apply it, and
`secure_delete` plus a WAL truncate after every delete keep a deleted note out
of freelist pages and the WAL.

A per-key `generation` is bumped by every write, including the operator CLI
running in another process, and `purges` by every delete. A chunk's apply is
refused if either moved since the chunk read the facts, so nothing the
operator removed comes back. The READ side uses `PRAGMA data_version` to
notice writes made by any other connection and reload its cache.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .memory_facts import AcceptedAdd, Fact, StagedNote
from .secrets import SecretsStore

log = logging.getLogger(__name__)

Key = tuple[str, str]
# One chunk never mixes persons or sessions: (room_id, person, session_id).
Group = tuple[str, str, str]

NOTES_KEY_SCOPE = "speaker_memory"
NOTES_KEY_NAME = "notes-key"
EXPIRED_COUNTER = "notes_expired"

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
CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    utterance_id TEXT NOT NULL,
    PRIMARY KEY (fact_id, session_id, utterance_id)
);
CREATE INDEX IF NOT EXISTS fact_sources_session ON fact_sources(session_id);
CREATE TABLE IF NOT EXISTS speaker_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    person TEXT NOT NULL,
    session_id TEXT NOT NULL,
    utterance_id TEXT NOT NULL,
    user_text BLOB NOT NULL,
    reply BLOB,
    untrusted INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS speaker_notes_group ON speaker_notes(room_id, person, session_id, id);
CREATE TABLE IF NOT EXISTS memory_batch_runs (
    run_date TEXT PRIMARY KEY,
    finished_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_day_counts (
    person TEXT NOT NULL,
    run_date TEXT NOT NULL,
    adds INTEGER NOT NULL DEFAULT 0,
    drops INTEGER NOT NULL DEFAULT 0,
    untrusted_adds INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (person, run_date)
);
CREATE TABLE IF NOT EXISTS memory_counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""


class StaleGeneration(Exception):
    """The speaker's facts changed since the chunk read them."""


@dataclass(frozen=True)
class Chunk:
    """The notes one writer call sees, and the ids its apply will delete.
    `ids` also holds notes that no longer decrypt, so they are consumed
    rather than retried forever."""

    group: Group
    notes: list[StagedNote]
    ids: tuple[int, ...]


@dataclass(frozen=True)
class ChunkWrite:
    group: Group
    run_date: str
    note_ids: tuple[int, ...]
    expected: tuple[int, int]
    adds: tuple[AcceptedAdd, ...] = ()
    drop_ids: tuple[int, ...] = ()
    reaffirmed: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def key(self) -> Key:
        return self.group[0], self.group[1]


@dataclass(frozen=True)
class DayCounts:
    adds: int = 0
    drops: int = 0
    untrusted_adds: int = 0


@dataclass(frozen=True)
class Removed:
    facts: int
    notes: int


def notes_cipher(secrets: SecretsStore) -> Fernet | None:
    """The staged-notes key, created on first use. None, and nothing is
    staged, when the keyring cannot give back a usable key."""
    try:
        key = secrets.get(NOTES_KEY_SCOPE, NOTES_KEY_NAME)
        if key is None:
            secrets.set(NOTES_KEY_SCOPE, NOTES_KEY_NAME, Fernet.generate_key().decode("ascii"))
            key = secrets.get(NOTES_KEY_SCOPE, NOTES_KEY_NAME)
        return Fernet(key) if key is not None else None
    except Exception as e:
        log.warning("speaker memory: notes key unusable (%s)", type(e).__name__)
        return None


class SpeakerMemoryStore:
    def __init__(
        self, path: Path, *, max_facts: int = 30, cipher: Fernet | None = None
    ) -> None:
        self._path = Path(path)
        self._max_facts = max_facts
        self._cipher = cipher
        self._warned_no_key = False
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._cache: dict[Key, list[Fact]] = {}
        self._seen_version: int | None = None
        self._closed = True
        self.open()

    def open(self) -> None:
        """(Re)connect. A no-op while open; after `close` it lets the server
        lifespan start again on the same app, as every TestClient does. Runs
        at lifespan start, before any staging or batch thread exists."""
        if not self._closed:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write = self._connect()
        self._write.executescript(_SCHEMA)
        _restrict_to_owner(self._path)
        self._read = self._connect()
        self._cache.clear()
        self._seen_version = None
        self._closed = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("PRAGMA secure_delete=ON")
        return conn

    def set_cipher(self, cipher: Fernet | None) -> None:
        self._cipher = cipher

    def can_read(self) -> bool:
        """A batch without the key would consume every note as unreadable."""
        return self._cipher is not None

    def close(self) -> None:
        with self._read_lock, self._write_lock:
            if self._closed:
                return
            self._read.close()
            self._write.close()
            self._closed = True

    # ---- facts: the turn-time read ----------------------------------------

    def facts(self, room_id: str, speaker_id: str) -> list[Fact]:
        """Newest first. Served from cache; a write by any connection since the
        last call (the writer thread, the CLI) reloads it first."""
        key = (room_id, speaker_id)
        with self._read_lock:
            self._refresh_if_changed()
            if key not in self._cache:
                self._cache[key] = self._load(key)
            return list(self._cache[key])

    def stamp(self, room_id: str, speaker_id: str) -> tuple[int, int]:
        """(generation, purges). Read this BEFORE the facts a write is based
        on: a write landing in between makes it stale and the write is refused."""
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

    # ---- staging ------------------------------------------------------------

    def stage_note(
        self,
        group: Group,
        utterance_id: str,
        note: StagedNote,
        *,
        now: float | None = None,
    ) -> bool:
        """Encrypt and keep one committed turn for tonight's batch. False,
        and nothing written, without a notes key. Blocking: call it off the loop."""
        cipher = self._cipher
        if cipher is None:
            self._warn_no_key_once()
            return False
        reply = cipher.encrypt(note.reply.encode("utf-8")) if note.reply else None
        row = (
            *group, utterance_id, cipher.encrypt(note.user_text.encode("utf-8")),
            reply, int(note.untrusted), time.time() if now is None else now,
        )
        with self._write_lock:
            if self._closed:
                log.info("speaker memory: store closed, note not staged")
                return False
            self._write.execute(
                "INSERT INTO speaker_notes (room_id, person, session_id, utterance_id, "
                "user_text, reply, untrusted, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        return True

    def _warn_no_key_once(self) -> None:
        if not self._warned_no_key:
            self._warned_no_key = True
            log.warning("speaker memory: no notes key; turns are not being staged")

    def expire_notes(self, older_than: float) -> int:
        """Delete notes staged before `older_than` (wall clock), unprocessed.
        Every loss is logged as a count and added to a persisted counter."""
        with self._write_lock:
            conn = self._write
            with _transaction(conn):
                count = conn.execute(
                    "DELETE FROM speaker_notes WHERE created_at < ?", (older_than,)
                ).rowcount
                if count:
                    _bump_counter(conn, EXPIRED_COUNTER, count)
            if count:
                _truncate_wal(conn)
        if count:
            log.warning("speaker memory: %d staged notes expired unprocessed", count)
        return count

    def counter(self, name: str) -> int:
        with self._read_lock:
            row = self._read.execute(
                "SELECT value FROM memory_counters WHERE name=?", (name,)
            ).fetchone()
        return row[0] if row else 0

    def staged_count(self) -> int:
        with self._read_lock:
            return self._read.execute("SELECT COUNT(*) FROM speaker_notes").fetchone()[0]

    # ---- the batch's reads --------------------------------------------------

    def pending_groups(self) -> list[Group]:
        """Every (room, person, session) with staged notes, oldest first."""
        with self._read_lock:
            rows = self._read.execute(
                "SELECT room_id, person, session_id FROM speaker_notes "
                "GROUP BY room_id, person, session_id ORDER BY MIN(id)"
            ).fetchall()
        return [tuple(r) for r in rows]

    def read_chunk(self, group: Group, limit: int) -> Chunk:
        with self._read_lock:
            rows = self._read.execute(
                "SELECT id, user_text, reply, untrusted FROM speaker_notes "
                "WHERE room_id=? AND person=? AND session_id=? ORDER BY id LIMIT ?",
                (*group, limit),
            ).fetchall()
        notes = [n for n in (self._decrypt(r) for r in rows) if n is not None]
        return Chunk(group=group, notes=notes, ids=tuple(r[0] for r in rows))

    def _decrypt(self, row: tuple) -> StagedNote | None:
        note_id, user_text, reply, untrusted = row
        if self._cipher is None:
            return None
        try:
            return StagedNote(
                id=note_id,
                user_text=self._cipher.decrypt(user_text).decode("utf-8"),
                reply=self._cipher.decrypt(reply).decode("utf-8") if reply else None,
                untrusted=bool(untrusted),
            )
        except (InvalidToken, UnicodeDecodeError):
            log.warning("speaker memory: staged note %d does not decrypt; consuming it", note_id)
            return None

    def day_counts(self, person: str, run_date: str) -> DayCounts:
        """Per PERSON per day: memory-enabled rooms share one budget."""
        with self._read_lock:
            row = self._read.execute(
                "SELECT adds, drops, untrusted_adds FROM memory_day_counts "
                "WHERE person=? AND run_date=?",
                (person, run_date),
            ).fetchone()
        return DayCounts(*row) if row else DayCounts()

    def ran_on(self, run_date: str) -> bool:
        with self._read_lock:
            return self._read.execute(
                "SELECT 1 FROM memory_batch_runs WHERE run_date=?", (run_date,)
            ).fetchone() is not None

    def record_run(self, run_date: str) -> None:
        with self._write_lock:
            self._write.execute(
                "INSERT OR REPLACE INTO memory_batch_runs VALUES (?, ?)",
                (run_date, time.time()),
            )

    # ---- the apply transaction ------------------------------------------------

    async def apply_chunk(
        self, write: ChunkWrite, is_enrolled: Callable[[str], bool]
    ) -> str:
        return await asyncio.to_thread(self.apply_chunk_sync, write, is_enrolled)

    def apply_chunk_sync(
        self, write: ChunkWrite, is_enrolled: Callable[[str], bool]
    ) -> str:
        """"applied", "forgotten" or "unenrolled"; raises StaleGeneration.
        The chunk's notes are deleted by their snapshotted ids, in the same
        transaction as the facts they produced. Enrollment is read before the
        lock, so a slow voiceprint read never holds up staging; it raises when
        enrollment is unknown, and the notes are kept."""
        enrolled = is_enrolled(write.group[1])
        with self._write_lock:
            conn = self._write
            with _transaction(conn):
                outcome = self._apply_in_transaction(conn, write, enrolled)
            _truncate_wal(conn)
        return outcome

    def _apply_in_transaction(
        self, conn: sqlite3.Connection, write: ChunkWrite,
        enrolled: bool,
    ) -> str:
        _room_id, person, session_id = write.group
        if _forgotten(conn, session_id):
            _delete_notes(conn, "session_id=?", (session_id,))
            return "forgotten"
        if not enrolled:
            _delete_notes(conn, "person=?", (person,))
            return "unenrolled"
        if _stamp(conn, write.key) != write.expected:
            raise StaleGeneration(write.key)
        utterances = _utterance_ids(conn, write)
        if write.drop_ids:
            _delete_facts(conn, _ids_in(write.drop_ids, "room_id=? AND speaker_id=? AND id"),
                          (*write.key, *write.drop_ids))
        self._insert_facts(conn, write, utterances)
        self._evict_over_cap(conn, write.key)
        if write.note_ids:
            _delete_notes(conn, _ids_in(write.note_ids, "room_id=? AND person=? AND session_id=? AND id"),
                          (*write.group, *write.note_ids))
        _bump_day(conn, write)
        _bump_generation(conn, write.key)
        return "applied"

    def _insert_facts(
        self, conn: sqlite3.Connection, write: ChunkWrite, utterances: dict[int, str]
    ) -> None:
        now = time.time()
        session_id = write.group[2]
        for add in write.adds:
            cited = [utterances[i] for i in add.note_ids if i in utterances]
            fact_id = conn.execute(
                "INSERT INTO speaker_facts (room_id, speaker_id, text, "
                "source_session, source_utterance, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (*write.key, add.text, session_id, " ".join(cited), now, now),
            ).lastrowid
            _link_sources(conn, fact_id, session_id, cited)
        for fact_id, note_ids in write.reaffirmed.items():
            touched = conn.execute(
                "UPDATE speaker_facts SET last_seen_at=? WHERE id=? AND room_id=? AND speaker_id=?",
                (now, fact_id, *write.key),
            ).rowcount
            if touched:
                _link_sources(conn, fact_id, session_id, [utterances[i] for i in note_ids if i in utterances])

    def _evict_over_cap(self, conn: sqlite3.Connection, key: Key) -> None:
        _delete_facts(
            conn,
            "room_id=? AND speaker_id=? AND id NOT IN ("
            "SELECT id FROM speaker_facts WHERE room_id=? AND speaker_id=? "
            "ORDER BY last_seen_at DESC, id DESC LIMIT ?)",
            (*key, *key, self._max_facts),
        )

    # ---- operator deletes ------------------------------------------------------

    def list_all(self) -> list[tuple[int, str, str, str, str]]:
        with self._read_lock:
            return self._read.execute(
                "SELECT id, room_id, speaker_id, text, source_session FROM speaker_facts "
                "ORDER BY room_id, speaker_id, last_seen_at DESC"
            ).fetchall()

    def forget_fact(self, fact_id: int) -> Removed:
        """The fact, plus any still-staged note it cites, or the next run
        would derive it again."""
        return self._purge(
            facts=("id=?", (fact_id,)),
            notes=(
                "utterance_id IN (SELECT utterance_id FROM fact_sources WHERE fact_id=?)",
                (fact_id,),
            ),
        )

    def forget_session(self, session_id: str) -> Removed:
        """Every fact ANY note of this session backed, and its staged notes."""
        return self._purge(
            facts=(
                "source_session=? OR id IN (SELECT fact_id FROM fact_sources WHERE session_id=?)",
                (session_id, session_id),
            ),
            notes=("session_id=?", (session_id,)),
            forget_session=session_id,
        )

    def wipe(self, room_id: str, speaker_id: str) -> Removed:
        return self._purge(
            facts=("room_id=? AND speaker_id=?", (room_id, speaker_id)),
            notes=("room_id=? AND person=?", (room_id, speaker_id)),
            purge_key=(room_id, speaker_id),
        )

    def _purge(
        self,
        *,
        facts: tuple[str, tuple],
        notes: tuple[str, tuple],
        forget_session: str | None = None,
        purge_key: Key | None = None,
    ) -> Removed:
        with self._write_lock:
            conn = self._write
            with _transaction(conn):
                keys = _keys_touched(conn, facts, notes) | ({purge_key} if purge_key else set())
                note_count = _delete_notes(conn, *notes)
                fact_count = _delete_facts(conn, *facts)
                for key in keys:
                    _bump_generation(conn, key, purge=True)
                if forget_session is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO speaker_forgotten_sessions VALUES (?)",
                        (forget_session,),
                    )
            _truncate_wal(conn)
        return Removed(facts=fact_count, notes=note_count)


class _transaction:
    """BEGIN IMMEDIATE ... COMMIT, rolled back on any exception."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.execute("BEGIN IMMEDIATE")
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> None:
        self._conn.execute("ROLLBACK" if exc_type is not None else "COMMIT")


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


def _bump_day(conn: sqlite3.Connection, write: ChunkWrite) -> None:
    conn.execute(
        "INSERT INTO memory_day_counts (person, run_date, adds, drops, untrusted_adds) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(person, run_date) DO UPDATE SET "
        "adds = adds + excluded.adds, drops = drops + excluded.drops, "
        "untrusted_adds = untrusted_adds + excluded.untrusted_adds",
        (write.group[1], write.run_date, len(write.adds), len(write.drop_ids),
         sum(a.untrusted for a in write.adds)),
    )


def _bump_counter(conn: sqlite3.Connection, name: str, by: int) -> None:
    conn.execute(
        "INSERT INTO memory_counters VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
        (name, by),
    )


def _ids_in(ids: tuple[int, ...], prefix: str) -> str:
    """`<prefix> IN (?, ?, ...)`. Callers skip an empty id list."""
    return f"{prefix} IN ({', '.join('?' * len(ids))})"


def _utterance_ids(conn: sqlite3.Connection, write: ChunkWrite) -> dict[int, str]:
    """Cited note id -> utterance id, read before the notes are deleted."""
    cited = {i for add in write.adds for i in add.note_ids}
    cited |= {i for ids in write.reaffirmed.values() for i in ids}
    ids = tuple(sorted(cited))
    if not ids:
        return {}
    rows = conn.execute(
        f"SELECT id, utterance_id FROM speaker_notes WHERE "
        f"{_ids_in(ids, 'room_id=? AND person=? AND session_id=? AND id')}",
        (*write.group, *ids),
    ).fetchall()
    return dict(rows)


def _link_sources(
    conn: sqlite3.Connection, fact_id: int, session_id: str, utterances: list[str]
) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO fact_sources VALUES (?, ?, ?)",
        [(fact_id, session_id, u) for u in utterances or [""]],
    )


def _keys_touched(
    conn: sqlite3.Connection, facts: tuple[str, tuple], notes: tuple[str, tuple]
) -> set[Key]:
    rows = conn.execute(
        f"SELECT DISTINCT room_id, speaker_id FROM speaker_facts WHERE {facts[0]}", facts[1]
    ).fetchall()
    rows += conn.execute(
        f"SELECT DISTINCT room_id, person FROM speaker_notes WHERE {notes[0]}", notes[1]
    ).fetchall()
    return {tuple(r) for r in rows}


def _delete_facts(conn: sqlite3.Connection, where: str, args: tuple) -> int:
    ids = [r[0] for r in conn.execute(f"SELECT id FROM speaker_facts WHERE {where}", args)]
    if not ids:
        return 0
    marks = ", ".join("?" * len(ids))
    conn.execute(f"DELETE FROM fact_sources WHERE fact_id IN ({marks})", ids)
    return conn.execute(f"DELETE FROM speaker_facts WHERE id IN ({marks})", ids).rowcount


def _delete_notes(conn: sqlite3.Connection, where: str, args: tuple) -> int:
    return conn.execute(f"DELETE FROM speaker_notes WHERE {where}", args).rowcount


def _truncate_wal(conn: sqlite3.Connection) -> None:
    """After a delete, so the deleted rows do not live on in the WAL."""
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _restrict_to_owner(path: Path) -> None:
    for target in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not target.exists():
            continue
        try:
            os.chmod(target, 0o600)
        except OSError:
            log.warning("speaker memory: could not restrict file mode on %s", target.name)
