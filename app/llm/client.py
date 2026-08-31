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
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


class LLMCallError(Exception):
    pass


_RETRYABLE = (APITimeoutError, RateLimitError, APIError)


def _repair_json(raw: str) -> str:
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
