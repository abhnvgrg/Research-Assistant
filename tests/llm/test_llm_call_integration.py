from __future__ import annotations

import app.graph.nodes as nodes
import app.llm.client as client_module
from app.graph.nodes import (
    MODEL_DECOMPOSER,
    MODEL_GRADER,
    MODEL_REFLECTION,
    MODEL_SYNTHESIZER,
    _call_decomposer_llm,
    _call_grader_llm,
    _call_reflection_llm,
    _call_synthesizer_llm,
)
from tests.conftest import make_chunk
from tests.llm.fakes import fake_completion


class RecordingCompletions:
    def __init__(self, response_content: str, total_tokens: int | None = None):
        self._response_content = response_content
        self._total_tokens = total_tokens
        self.last_kwargs: dict = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return fake_completion(self._response_content, total_tokens=self._total_tokens)


class RecordingClient:
    def __init__(self, response_content: str, total_tokens: int | None = None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = RecordingCompletions(response_content, total_tokens)


def _patch(monkeypatch, response_content: str, total_tokens: int | None = None) -> RecordingClient:
    fake_client = RecordingClient(response_content, total_tokens)
    monkeypatch.setattr(client_module, "get_client", lambda: fake_client)
    return fake_client


async def test_decomposer_llm_uses_correct_model_and_includes_query(monkeypatch):
    fake = _patch(monkeypatch, '{"topic_identified": true, "sub_questions": []}', total_tokens=310)

    result, tokens = await _call_decomposer_llm("explain backprop")

    assert fake.chat.completions.last_kwargs["model"] == MODEL_DECOMPOSER
    messages = fake.chat.completions.last_kwargs["messages"]
    assert "explain backprop" in messages[1]["content"]
    assert result["topic_identified"] is True
    assert tokens == 310


async def test_grader_llm_truncates_chunk_to_500_chars(monkeypatch):
    fake = _patch(monkeypatch, '{"relevant": true, "score": 0.9, "reason": "ok"}')
    long_chunk = make_chunk(text="z" * 1000, source="https://source.example/doc")

    await _call_grader_llm(long_chunk, "some query")

    assert fake.chat.completions.last_kwargs["model"] == MODEL_GRADER
    user_message = fake.chat.completions.last_kwargs["messages"][1]["content"]
    assert user_message.count("z") == 500


async def test_synthesizer_llm_wraps_chunks_in_xml_tags(monkeypatch):
    fake = _patch(monkeypatch, "Answer text with [1] citation.", total_tokens=610)
    graded = [
        {**make_chunk("chunk text here", source="https://a.com"), "relevant": True, "grade_score": 0.9, "grade_reason": "ok"}
    ]

    result = await _call_synthesizer_llm("query", graded)

    assert fake.chat.completions.last_kwargs["model"] == MODEL_SYNTHESIZER
    user_message = fake.chat.completions.last_kwargs["messages"][1]["content"]
    assert '<chunk index="1" source="https://a.com">chunk text here</chunk>' in user_message
    assert result["citations"] == {"1": "https://a.com"}
    assert result["tokens_used"] == 610


async def test_synthesizer_llm_strips_hallucinated_citation_end_to_end(monkeypatch):
    fake = _patch(monkeypatch, "Fact [1] and a fabricated fact [9].")
    graded = [{**make_chunk(source="https://only-source.com"), "relevant": True, "grade_score": 0.9, "grade_reason": "ok"}]

    result = await _call_synthesizer_llm("query", graded)

    assert "[9]" not in result["answer"]
    assert "9" not in result["citations"]
    assert result["citations"] == {"1": "https://only-source.com"}


async def test_reflection_llm_includes_answer_and_subquestions(monkeypatch):
    fake = _patch(monkeypatch, '{"pass_": true, "checklist": {}, "gap": "nothing"}', total_tokens=95)
    sub_qs = [{"question": "What is X?", "intent": "general", "aliases": []}]

    result, tokens = await _call_reflection_llm("original query", sub_qs, "the synthesized answer")

    assert fake.chat.completions.last_kwargs["model"] == MODEL_REFLECTION
    user_message = fake.chat.completions.last_kwargs["messages"][1]["content"]
    assert "What is X?" in user_message
    assert "the synthesized answer" in user_message
    assert result["pass_"] is True
    assert tokens == 95
