"""
Shared LLM client — every real API call in nodes.py goes through
call_json() or call_text() below, never through the OpenAI SDK
directly. This is the single seam where retry policy, model
selection, and JSON repair live, so every node gets them for free.

Design decisions this file encodes (from our earlier scrutiny):
  - Retry 3x with exponential backoff on transient failures (5xx,
    timeouts, rate limits) — never on 4xx (bad request won't fix
    itself by retrying).
  - json.loads() first, then a light repair pass (strip markdown
    fences, fix trailing commas) before giving up — LLMs asked for
    "JSON only" sometimes still wrap it in ```json fences or add a
    trailing comma under load.
  - Model name is a parameter, not hardcoded, so node-level model
    selection (gpt-4o-mini for grading/decomp/reflection, gpt-4o for
    synthesis) is explicit at every call site.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Lazily constructed singleton — avoids creating a client at
    import time (breaks tests that don't set OPENAI_API_KEY)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


class LLMCallError(Exception):
    """Raised after all retries are exhausted, or on unrecoverable
    JSON parse failure. Nodes catch this and write to state['error']."""


_RETRYABLE = (APITimeoutError, RateLimitError, APIError)


def _repair_json(raw: str) -> str:
    """Handles the two most common ways an LLM violates 'JSON only':
    wrapping the object in markdown fences, or leaving a trailing
    comma before a closing bracket."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text.strip()


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
async def _create_completion(
    *, model: str, system_prompt: str, user_prompt: str, json_mode: bool
):
    client = get_client()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return await client.chat.completions.create(**kwargs)


async def call_json(*, model: str, system_prompt: str, user_prompt: str) -> tuple[dict, int]:
    """Calls the LLM expecting a JSON object back. Retries on
    transient API errors, then attempts a repair pass on the raw
    text before giving up. Raises LLMCallError on total failure —
    nodes must catch this. Returns (parsed_json, tokens_used) — the
    real total_tokens from the OpenAI response's usage field, used to
    track per-run cost against each user's quota (see
    app/quota_store.py). Defaults to 0 if the response genuinely has
    no usage info (defensive — should not happen against the real
    API, but a fake/mocked response in a test might omit it)."""
    try:
        response = await _create_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )
    except _RETRYABLE as e:
        raise LLMCallError(f"LLM call failed after retries: {e}") from e

    raw = response.choices[0].message.content or ""
    tokens_used = _extract_total_tokens(response)

    try:
        return json.loads(raw), tokens_used
    except json.JSONDecodeError:
        logger.warning("First-pass JSON parse failed, attempting repair: %r", raw[:200])
        try:
            return json.loads(_repair_json(raw)), tokens_used
        except json.JSONDecodeError as e:
            raise LLMCallError(f"Could not parse LLM JSON output even after repair: {e}") from e


def _extract_total_tokens(response) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, "total_tokens", 0) or 0


async def call_text(*, model: str, system_prompt: str, user_prompt: str) -> tuple[str, int]:
    """Calls the LLM expecting free-form prose (Markdown) back —
    used by the synthesizer, where forcing a JSON schema on long
    prose causes truncation/compression artifacts. Returns
    (text, tokens_used) — same real-usage-tracking contract as
    call_json()."""
    try:
        response = await _create_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=False,
        )
    except _RETRYABLE as e:
        raise LLMCallError(f"LLM call failed after retries: {e}") from e

    return response.choices[0].message.content or "", _extract_total_tokens(response)
