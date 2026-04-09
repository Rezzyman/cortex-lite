"""
Embedding providers. Tries Ollama (local, free) first,
then Voyage, then OpenAI.
"""

import os

import httpx


def _get_provider() -> str:
    provider = os.environ.get("CORTEX_EMBEDDING_PROVIDER", "").lower()
    if provider:
        return provider
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def embed_dim() -> int:
    """Return the embedding dimension for the active provider/model."""
    provider = _get_provider()
    if provider == "ollama":
        model = os.environ.get("CORTEX_EMBEDDING_MODEL", "mxbai-embed-large")
        # Known dimensions for common models
        dims = {
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }
        return dims.get(model, 1024)
    if provider == "voyage":
        return 1024
    if provider == "openai":
        return 1536
    return 1024


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts for storage."""
    provider = _get_provider()
    if provider == "voyage":
        return _voyage(texts, "document")
    if provider == "openai":
        return _openai(texts)
    return _ollama(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query for search."""
    provider = _get_provider()
    if provider == "voyage":
        return _voyage([text], "query")[0]
    if provider == "openai":
        return _openai([text])[0]
    return _ollama([text])[0]


# -- providers ---------------------------------------------------------------

def _ollama(texts: list[str]) -> list[list[float]]:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("CORTEX_EMBEDDING_MODEL", "mxbai-embed-large")
    out: list[list[float]] = []
    for text in texts:
        resp = httpx.post(
            f"{url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        out.append(resp.json()["embedding"])
    return out


def _voyage(texts: list[str], input_type: str = "document") -> list[list[float]]:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    all_emb: list[list[float]] = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "voyage-3", "input": batch, "input_type": input_type},
            timeout=30,
        )
        resp.raise_for_status()
        all_emb.extend([d["embedding"] for d in resp.json()["data"]])
    return all_emb


def _openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]
