"""
CORTEX Lite Search — Hybrid 5-factor scoring.

score = 0.55 * cosine_similarity
      + 0.20 * text_match
      + 0.15 * recency
      + 0.05 * resonance
      + 0.05 * priority_boost

Brute-force cosine over numpy. Fast to ~10K memories.
"""

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from cortex_lite.chunker import count_tokens
from cortex_lite.db import _deserialize_vec, resolve_agent
from cortex_lite.embeddings import embed_query


@dataclass
class SearchResult:
    id: int
    content: str
    source: str | None
    score: float
    entities: list[str]
    semantic_tags: list[str]


def search(conn, query: str, agent_id: str = "default", limit: int = 10) -> list[SearchResult]:
    """
    Hybrid search combining vector similarity, keyword match,
    recency, resonance, and priority.
    """
    agent_num = resolve_agent(conn, agent_id)
    query_vec = np.array(embed_query(query), dtype=np.float32)
    query_lower = query.lower()

    rows = conn.execute(
        """SELECT id, content, source, embedding, entities, semantic_tags,
                  priority, resonance_score, created_at
           FROM memory_nodes
           WHERE agent_id = ? AND status = 'active' AND embedding IS NOT NULL""",
        (agent_num,),
    ).fetchall()

    if not rows:
        return []

    now = time.time()
    scored: list[tuple[float, dict]] = []

    for row in rows:
        emb = np.array(_deserialize_vec(row["embedding"]), dtype=np.float32)

        # 1. Cosine similarity
        dot = float(np.dot(query_vec, emb))
        norm = float(np.linalg.norm(query_vec) * np.linalg.norm(emb))
        cosine = dot / norm if norm > 0 else 0.0

        # 2. Text match (case-insensitive substring)
        text_match = 1.0 if query_lower in row["content"].lower() else 0.0

        # 3. Recency (30-day half-life exponential decay)
        try:
            from datetime import datetime
            created = datetime.fromisoformat(row["created_at"]).timestamp()
            age_days = (now - created) / 86400
        except Exception:
            age_days = 30.0
        recency = math.exp(-0.023 * age_days)

        # 4. Resonance (normalized to 0-1)
        resonance = min((row["resonance_score"] or 5.0) / 10.0, 1.0)

        # 5. Priority boost
        prio_map = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.3, 4: 0.1}
        priority_boost = prio_map.get(row["priority"], 0.5)

        score = (
            0.55 * cosine
            + 0.20 * text_match
            + 0.15 * recency
            + 0.05 * resonance
            + 0.05 * priority_boost
        )

        scored.append((score, dict(row)))

    scored.sort(key=lambda x: -x[0])
    top = scored[:limit]

    # Update access counts
    if top:
        ids = [r["id"] for _, r in top]
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"""UPDATE memory_nodes
                SET access_count = access_count + 1,
                    last_accessed_at = datetime('now')
                WHERE id IN ({placeholders})""",
            ids,
        )
        conn.commit()

    return [
        SearchResult(
            id=r["id"],
            content=r["content"],
            source=r["source"],
            score=s,
            entities=json.loads(r["entities"]) if r["entities"] else [],
            semantic_tags=json.loads(r["semantic_tags"]) if r["semantic_tags"] else [],
        )
        for s, r in top
    ]


def recall(conn, query: str, agent_id: str = "default", token_budget: int = 4000) -> str:
    """
    Token-budget-aware context retrieval.

    Returns the most relevant memories that fit within the budget,
    formatted as a single string.
    """
    results = search(conn, query, agent_id=agent_id, limit=50)

    parts: list[str] = []
    used = 0

    for r in results:
        tokens = count_tokens(r.content)
        if used + tokens > token_budget:
            break
        parts.append(r.content)
        used += tokens

    return "\n\n".join(parts)
