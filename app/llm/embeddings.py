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
    if not texts:
        return []

    client = get_client()
    try:
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    except _RETRYABLE as e:
        raise LLMCallError(f"Embedding call failed after retries: {e}") from e

    return [item.embedding for item in response.data]


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    return vectors[0]
