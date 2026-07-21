"""
Tests for app.llm.client.call_json — mocks OpenAI's client.chat.
completions.create directly (not our own wrapper), so these tests
prove the JSON repair and retry logic work against realistic raw
API response text, not just against our own assumptions.
"""

from __future__ import annotations

import pytest
from openai import APITimeoutError, RateLimitError

import app.llm.client as client_module
from app.llm.client import LLMCallError, call_json
from tests.llm.fakes import fake_completion


class FakeCompletions:
    """Drop-in replacement for client.chat.completions with a
    scriptable sequence of responses/exceptions per call."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeClient:
    def __init__(self, responses: list):
        self.chat = FakeChat(FakeCompletions(responses))


def _patch_client(monkeypatch, fake_client: FakeClient):
    monkeypatch.setattr(client_module, "get_client", lambda: fake_client)


async def test_call_json_happy_path(monkeypatch):
    fake = FakeClient([fake_completion('{"topic_identified": true, "score": 0.9}')])
    _patch_client(monkeypatch, fake)

    result, tokens = await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert result == {"topic_identified": True, "score": 0.9}
    assert tokens == 0  # fake_completion with no usage attached defaults to 0


async def test_call_json_extracts_real_token_usage(monkeypatch):
    """Proves the actual token-counting path works — not just that
    it defaults to 0 when usage is absent. This is what makes the
    quota system meaningful: without this, quota_store.add_usage()
    would only ever be called with 0, making the quota check a no-op
    no matter what the user actually did."""
    fake = FakeClient([fake_completion('{"ok": true}', total_tokens=742)])
    _patch_client(monkeypatch, fake)

    _, tokens = await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert tokens == 742


async def test_call_json_repairs_markdown_fence(monkeypatch):
    """LLMs told 'JSON only' sometimes still wrap the object in
    ```json fences — the repair pass must strip these."""
    fake = FakeClient([fake_completion('```json\n{"ok": true}\n```')])
    _patch_client(monkeypatch, fake)

    result, _tokens = await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert result == {"ok": True}


async def test_call_json_repairs_trailing_comma(monkeypatch):
    fake = FakeClient([fake_completion('{"a": 1, "b": 2,}')])
    _patch_client(monkeypatch, fake)

    result, _tokens = await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert result == {"a": 1, "b": 2}


async def test_call_json_raises_llm_call_error_on_unrepairable_json(monkeypatch):
    fake = FakeClient([fake_completion("this is not json at all, sorry!")])
    _patch_client(monkeypatch, fake)

    with pytest.raises(LLMCallError, match="Could not parse"):
        await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")


async def test_call_json_retries_on_rate_limit_then_succeeds(monkeypatch):
    """Proves the tenacity retry wrapper actually retries transient
    errors instead of failing on the first attempt."""
    import httpx

    fake_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    rate_limit_error = RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=fake_request),
        body=None,
    )

    fake = FakeClient([rate_limit_error, fake_completion('{"ok": true}')])
    _patch_client(monkeypatch, fake)

    result, _tokens = await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert result == {"ok": True}
    assert fake.chat.completions.call_count == 2


async def test_call_json_gives_up_after_max_retries(monkeypatch):
    """3 consecutive timeouts should exhaust tenacity's stop_after_
    attempt(3) and surface as LLMCallError, not hang or crash the
    node with an unrelated exception type."""
    import httpx

    fake_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    timeout_error = APITimeoutError(request=fake_request)

    fake = FakeClient([timeout_error, timeout_error, timeout_error])
    _patch_client(monkeypatch, fake)

    with pytest.raises(LLMCallError, match="failed after retries"):
        await call_json(model="gpt-4o-mini", system_prompt="sys", user_prompt="usr")

    assert fake.chat.completions.call_count == 3
