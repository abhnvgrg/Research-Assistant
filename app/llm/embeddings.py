"""
Embedding client — wraps OpenAI's embeddings endpoint with the same
retry policy as app.llm.client, since embedding calls hit the same
rate limits and transient failure modes as chat completions.

Kept separate from client.py because embeddings return vectors, not
chat completions, and batching semantics differ (embed multiple
texts in one call, vs one prompt per chat call).
"""

from __future__ import annotations

from openai import APIError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.client import LLMCallError, get_client

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

_RETRYABLE = (APITimeoutError, RateLimitError, APIError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts in a single API call (up to 2048
    inputs per OpenAI's limit — well above our per-query sub-question
    count, so callers never need to chunk this themselves).

    Returns one embedding vector per input text, in the same order.
    """
    if not texts:
        return []

    client = get_client()
    try:
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    except _RETRYABLE as e:
        raise LLMCallError(f"Embedding call failed after retries: {e}") from e

    return [item.embedding for item in response.data]


async def embed_text(text: str) -> list[float]:
    """Single-text convenience wrapper around embed_texts."""
    vectors = await embed_texts([text])
    return vectors[0]
