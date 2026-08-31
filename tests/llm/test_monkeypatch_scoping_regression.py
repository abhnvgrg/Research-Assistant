from __future__ import annotations

import app.llm.client as client_module
import app.llm.embeddings as embeddings_module
from app.llm.embeddings import embed_text
from tests.llm.fakes import fake_completion


class FakeEmbeddingsClient:
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
    fake = FakeEmbeddingsClient()
    monkeypatch.setattr(client_module, "get_client", lambda: fake)

    import pytest
    from openai import OpenAIError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embeddings_module._client = None
    client_module._client = None

    with pytest.raises(OpenAIError, match="Missing credentials"):
        await embed_text("some text")


async def test_patching_embeddings_module_directly_works(monkeypatch):
    fake = FakeEmbeddingsClient()
    monkeypatch.setattr(embeddings_module, "get_client", lambda: fake)

    result = await embed_text("some text")

    assert result == [0.42]
