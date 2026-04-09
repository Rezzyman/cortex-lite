"""Token-aware text chunking."""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 256
CHUNK_OVERLAP = 25


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into token-aware chunks with overlap.

    Returns [{"text": str, "index": int, "tokens": int}, ...]
    """
    if not text.strip():
        return []

    tokens = _encoder.encode(text)
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append({
            "text": _encoder.decode(chunk_tokens),
            "index": idx,
            "tokens": len(chunk_tokens),
        })
        idx += 1
        start += chunk_size - overlap

    return chunks
