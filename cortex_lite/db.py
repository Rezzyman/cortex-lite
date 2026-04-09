"""SQLite storage backend for CORTEX Lite."""

import sqlite3
import struct
from pathlib import Path


def _serialize_vec(vec: list[float]) -> bytes:
    """Pack a float list into a compact binary blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vec(blob: bytes) -> list[float]:
    """Unpack a binary blob into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def connect(db_path: str = "cortex.db") -> sqlite3.Connection:
    """Open (and initialize) a CORTEX Lite database."""
    db_path = str(Path(db_path).expanduser())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT    NOT NULL UNIQUE,
            name        TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS memory_nodes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id        INTEGER NOT NULL REFERENCES agents(id),
            content         TEXT    NOT NULL,
            source          TEXT,
            chunk_index     INTEGER DEFAULT 0,
            embedding       BLOB,
            entities        TEXT    DEFAULT '[]',
            semantic_tags   TEXT    DEFAULT '[]',
            priority        INTEGER DEFAULT 2,
            resonance_score REAL    DEFAULT 5.0,
            access_count    INTEGER DEFAULT 0,
            last_accessed_at TEXT,
            status          TEXT    DEFAULT 'active',
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_mn_agent_status
            ON memory_nodes(agent_id, status);
        CREATE INDEX IF NOT EXISTS idx_mn_agent_priority
            ON memory_nodes(agent_id, status, priority);
    """)


def resolve_agent(conn: sqlite3.Connection, agent_id: str = "default") -> int:
    """Get or create an agent, return its integer ID."""
    row = conn.execute(
        "SELECT id FROM agents WHERE external_id = ?", (agent_id,)
    ).fetchone()
    if row:
        return row["id"]

    cur = conn.execute(
        "INSERT INTO agents (external_id, name) VALUES (?, ?)",
        (agent_id, agent_id.capitalize()),
    )
    conn.commit()
    return cur.lastrowid
