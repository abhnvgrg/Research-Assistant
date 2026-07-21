"""
Regression test for a real bug caught while building the full-stack
dry run: app.llm.embeddings does `from app.llm.client import
get_client`, which creates a SEPARATE local binding in the
embeddings module's namespace. Patching app.llm.client.get_client
alone does NOT affect calls made from within embeddings.py — you
must patch app.llm.embeddings.get_client too.

This test exists so that if someone "cleans up" embeddings.py to
share more code with client.py, or changes the import style, this
failure mode gets caught immediately rather than silently reappearing
only when someone tries to test embeddings.py in isolation.
"""

from __future__ import annotations

import app.llm.client as client_module
import app.llm.embeddings as embeddings_module
from app.llm.embeddings import embed_text
from tests.llm.fakes import fake_completion


class FakeEmbeddingsClient:
    """Only implements .embeddings.create — used to prove which
    get_client() binding actually gets called."""

    class embeddings:
        @staticmethod
        async def create(*, model, input):
            from openai.types import CreateEmbeddingResponse, Embedding
            from openai.types.create_embedding_response import Usage

            return CreateEmbeddingResponse(
                data=[Embedding(embedding=[0.42], index=0, object="embedding")],
                model=model,
                object="list",
                usage=Usage(prompt_tokens=1, total_tokens=1),
            )


async def test_patching_client_module_alone_does_not_affect_embeddings(monkeypatch):
    """Documents the gotcha directly: patching app.llm.client.
    get_client alone must NOT make embed_text() use the fake client,
    because embeddings.py imported its own local reference."""
    fake = FakeEmbeddingsClient()
    monkeypatch.setattr(client_module, "get_client", lambda: fake)

    # embeddings_module.get_client is still the REAL function here,
    # which would try to construct a real AsyncOpenAI client and
    # fail without OPENAI_API_KEY set. We catch that to prove the
    # patch on client_module alone had no effect.
    import pytest
    from openai import OpenAIError

    with pytest.raises(OpenAIError, match="Missing credentials"):
        await embed_text("some text")


async def test_patching_embeddings_module_directly_works(monkeypatch):
    """The correct pattern: patch the name where it's actually
    looked up (embeddings_module.get_client), not just its origin."""
    fake = FakeEmbeddingsClient()
    monkeypatch.setattr(embeddings_module, "get_client", lambda: fake)

    result = await embed_text("some text")

    assert result == [0.42]
