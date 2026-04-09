"""
CORTEX Lite — Lightweight cognitive memory for AI agents.

Zero-config, SQLite-backed, embeds locally via Ollama.
Designed as the accessible on-ramp to the full CORTEX memory system.

Usage:
    from cortex_lite import CortexLite

    cx = CortexLite()                          # creates ./cortex.db
    cx.store("The meeting is at 3pm Friday")
    results = cx.search("when is the meeting")
    context = cx.recall("meeting details", token_budget=2000)
"""

import json
from dataclasses import dataclass
from pathlib import Path

from cortex_lite.chunker import chunk_text, count_tokens
from cortex_lite.db import _serialize_vec, connect, resolve_agent
from cortex_lite.embeddings import embed_texts
from cortex_lite.search import SearchResult, recall, search


@dataclass
class StoreResult:
    memory_ids: list[int]
    chunks: int
    agent_id: int


class CortexLite:
    """
    Lightweight cognitive memory backed by SQLite.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. Created if it doesn't exist.
    agent : str
        Default agent namespace for memory isolation.
    """

    def __init__(self, db_path: str = "cortex.db", agent: str = "default"):
        self._conn = connect(db_path)
        self._agent = agent

    # -- core API -------------------------------------------------------------

    def store(
        self,
        content: str,
        source: str = "api",
        priority: int = 2,
        agent: str | None = None,
    ) -> StoreResult:
        """
        Ingest text into memory. Chunks, embeds, and stores.

        Parameters
        ----------
        content : str   The text to remember.
        source : str    Where it came from (for provenance).
        priority : int  0 = critical, 4 = ephemeral.
        agent : str     Override the default agent namespace.
        """
        agent_id = agent or self._agent
        agent_num = resolve_agent(self._conn, agent_id)
        chunks = chunk_text(content)

        if not chunks:
            return StoreResult(memory_ids=[], chunks=0, agent_id=agent_num)

        embeddings = embed_texts([c["text"] for c in chunks])
        ids: list[int] = []

        for i, chunk in enumerate(chunks):
            emb_blob = _serialize_vec(embeddings[i])
            cur = self._conn.execute(
                """INSERT INTO memory_nodes
                   (agent_id, content, source, chunk_index,
                    embedding, priority, resonance_score, status)
                   VALUES (?, ?, ?, ?, ?, ?, 5.0, 'active')""",
                (agent_num, chunk["text"], source, i, emb_blob, priority),
            )
            ids.append(cur.lastrowid)

        self._conn.commit()
        return StoreResult(memory_ids=ids, chunks=len(chunks), agent_id=agent_num)

    def store_file(
        self,
        path: str,
        priority: int = 2,
        agent: str | None = None,
    ) -> StoreResult:
        """Ingest a text or markdown file."""
        content = Path(path).expanduser().read_text()
        return self.store(content, source=str(path), priority=priority, agent=agent)

    def search(
        self,
        query: str,
        limit: int = 10,
        agent: str | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector similarity + keyword + recency + resonance + priority.
        """
        return search(self._conn, query, agent_id=agent or self._agent, limit=limit)

    def recall(
        self,
        query: str,
        token_budget: int = 4000,
        agent: str | None = None,
    ) -> str:
        """
        Token-budgeted recall. Returns the most relevant memories
        that fit within the budget as a single context string.
        """
        return recall(
            self._conn, query, agent_id=agent or self._agent, token_budget=token_budget
        )

    def forget(self, memory_id: int) -> None:
        """Soft-delete a memory by ID."""
        self._conn.execute(
            "UPDATE memory_nodes SET status = 'deleted', updated_at = datetime('now') WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def prune(self, max_age_days: int = 90, min_resonance: float = 2.0) -> int:
        """
        Remove old, low-value memories. Returns count of pruned nodes.

        This is CORTEX Lite's simplified alternative to the full
        dream cycle. No consolidation, no REM — just cleanup.
        """
        cur = self._conn.execute(
            """UPDATE memory_nodes
               SET status = 'deleted', updated_at = datetime('now')
               WHERE status = 'active'
                 AND resonance_score < ?
                 AND julianday('now') - julianday(created_at) > ?
                 AND priority >= 3
               """,
            (min_resonance, max_age_days),
        )
        self._conn.commit()
        return cur.rowcount

    def status(self, agent: str | None = None) -> dict:
        """Return stats for the given agent."""
        agent_num = resolve_agent(self._conn, agent or self._agent)
        row = self._conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active,
                 SUM(CASE WHEN status = 'deleted' THEN 1 ELSE 0 END) AS deleted,
                 MIN(created_at) AS oldest,
                 MAX(created_at) AS newest
               FROM memory_nodes WHERE agent_id = ?""",
            (agent_num,),
        ).fetchone()
        return dict(row)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
