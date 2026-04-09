# CORTEX Lite

Lightweight cognitive memory for AI agents. Zero-config, SQLite-backed, embeds locally.

```
pip install cortex-lite
```

> **CORTEX Lite** is the accessible on-ramp to [CORTEX](https://github.com/Rezzyman/cortex) — the memory system that scored 500/500 on LongMemEval and 94.5% on CogBench. Same core memory loop. No Postgres. No infrastructure. One file.

---

## Quickstart (30 seconds)

**1. Install**

```bash
pip install cortex-lite
```

**2. Pull an embedding model** (free, runs locally)

```bash
ollama pull mxbai-embed-large
```

> Don't have Ollama? [Install it](https://ollama.com/download), or skip this and set an `OPENAI_API_KEY` instead.

**3. Use it**

```python
from cortex_lite import CortexLite

cx = CortexLite()  # creates cortex.db in current directory

# Store memories
cx.store("The client meeting is Friday at 2pm in Denver")
cx.store("Jeff prefers email over Slack for project updates")
cx.store("The API rate limit is 100 requests per minute")

# Search — returns ranked results by relevance
results = cx.search("when is the client meeting")
for r in results:
    print(f"[{r.score:.2f}] {r.content}")

# Recall — get context that fits in a token budget (for LLM prompts)
context = cx.recall("what do I know about Jeff", token_budget=2000)
print(context)
```

That's it. No database server. No API keys (if using Ollama). No config files.

---

## CLI

```bash
# Store
cortex-lite store "The deploy is scheduled for Thursday"

# Search
cortex-lite search "when is the deploy"

# Token-budgeted recall
cortex-lite recall "deployment schedule" --budget 2000

# Stats
cortex-lite status

# Cleanup old low-value memories
cortex-lite prune --max-age 90

# Use a specific database file
cortex-lite --db ~/my-agent/memory.db search "meeting notes"

# Use agent namespaces (isolate memories per agent)
cortex-lite --agent aria store "Content calendar is due Monday"
cortex-lite --agent arlo store "Run diagnostics at midnight"
```

---

## How it works

CORTEX Lite does three things:

1. **Store** — Chunks your text, converts it to embeddings (vectors), and saves it to a SQLite database.
2. **Search** — Finds the most relevant memories using a hybrid 5-factor scoring system.
3. **Recall** — Returns the best memories that fit within a token budget, ready to inject into an LLM prompt.

### Hybrid search scoring

Every search blends five signals:

| Factor | Weight | What it does |
|--------|--------|-------------|
| Cosine similarity | 55% | How semantically close is this memory to the query? |
| Text match | 20% | Does the query appear verbatim in the memory? |
| Recency | 15% | How recent is the memory? (30-day half-life) |
| Resonance | 5% | How important is this memory? (access frequency) |
| Priority | 5% | What priority tier was it stored at? |

This means CORTEX Lite doesn't just do keyword search (like grep) or just do vector search (like a pure embedding lookup). It blends both with recency and importance signals — the same approach used in the full CORTEX system.

### Where does the data live?

One file: `cortex.db` (SQLite). You can copy it, back it up, delete it, or move it to another machine. No server process. No ports. No credentials.

---

## Embedding providers

CORTEX Lite auto-detects your embedding provider:

| Provider | Setup | Cost |
|----------|-------|------|
| **Ollama** (default) | `ollama pull mxbai-embed-large` | Free (local) |
| **OpenAI** | Set `OPENAI_API_KEY` | ~$0.02 / 1M tokens |
| **Voyage** | Set `VOYAGE_API_KEY` | ~$0.06 / 1M tokens |

To override auto-detection:

```bash
export CORTEX_EMBEDDING_PROVIDER=ollama
export CORTEX_EMBEDDING_MODEL=mxbai-embed-large  # or nomic-embed-text, etc.
```

---

## Agent namespaces

Memories are isolated per agent. This lets multiple agents share one database without cross-contamination:

```python
cx = CortexLite("shared.db")

cx.store("Content calendar due Monday", agent="aria")
cx.store("Run diagnostics at midnight", agent="arlo")

# Only returns Aria's memories
cx.search("what's due this week", agent="aria")
```

---

## Python API

```python
from cortex_lite import CortexLite

cx = CortexLite(db_path="cortex.db", agent="default")

# Store text (chunks, embeds, and saves automatically)
result = cx.store(content, source="api", priority=2)
# → StoreResult(memory_ids=[1, 2], chunks=2, agent_id=1)

# Store a file
result = cx.store_file("notes.md", priority=1)

# Search (returns ranked SearchResult list)
results = cx.search(query, limit=10)
# → [SearchResult(id, content, source, score, entities, semantic_tags), ...]

# Recall (token-budgeted context string)
context = cx.recall(query, token_budget=4000)

# Soft-delete a memory
cx.forget(memory_id=42)

# Prune old low-value memories
pruned = cx.prune(max_age_days=90, min_resonance=2.0)

# Stats
stats = cx.status()
# → {"total": 150, "active": 142, "deleted": 8, "oldest": "...", "newest": "..."}
```

---

## CORTEX Lite vs CORTEX (Full)

| | CORTEX Lite | CORTEX |
|---|---|---|
| Storage | SQLite (one file) | PostgreSQL + pgvector |
| Setup | `pip install cortex-lite` | Postgres + extensions + migrations |
| Embeddings | Ollama / OpenAI / Voyage | Same |
| Search | 5-factor hybrid | 7-factor + HNSW index |
| Scale | 0–10K memories | 10K–100K+ |
| Consolidation | `prune()` | Full dream cycles (SWS + REM) |
| Hippocampal encoding | — | Dentate Gyrus sparse coding + CA1 novelty |
| Emotional valence | — | 6-dimensional |
| Procedural memory | — | Skill discovery + proficiency tracking |
| Benchmarks | — | 500/500 LongMemEval, 94.5% CogBench |

**CORTEX Lite is the starting point.** When you outgrow it — more memories, dream cycles, novelty detection — upgrade to [CORTEX](https://github.com/Rezzyman/cortex).

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) (for free local embeddings) or an OpenAI/Voyage API key

---

## License

MIT

Built by [ATERNA](https://aterna.ai)
