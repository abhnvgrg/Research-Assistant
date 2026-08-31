from __future__ import annotations

import asyncio
import json
import uuid

import app.llm.client as llm_client_module
import app.llm.embeddings as embeddings_module
import app.retrieval.pinecone_store as pinecone_module
import app.retrieval.tavily_search as tavily_module
from app.graph.build import research_graph
from app.graph.state import ResearchState
from tests.llm.fakes import fake_completion


class ScriptedOpenAIChat:
    def __init__(self):
        self.call_log: list[str] = []

    async def create(self, *, model, messages, **kwargs):
        system = messages[0]["content"]
        self.call_log.append(model)

        if "research query analyst" in system:
            return fake_completion(json.dumps({
                "topic_identified": True,
                "multi_intent": False,
                "recency_required": False,
                "corrected_query": "Explain how attention mechanisms work",
                "sub_questions": [
                    {"question": "What is an attention mechanism?", "intent": "general", "aliases": []},
                    {"question": "How is attention computed in transformers?", "intent": "general", "aliases": []},
                ],
            }), total_tokens=310)
        if "relevance grader" in system:
            return fake_completion(
                json.dumps({"relevant": True, "score": 0.85, "reason": "directly explains the mechanism"}),
                total_tokens=210,
            )
        if "quality reviewer" in system:
            return fake_completion(json.dumps({
                "pass_": True,
                "checklist": {"q1": True, "q2": True, "q3": False, "q4": "N/A"},
                "gap": "nothing",
            }), total_tokens=145)
        return fake_completion(
            "Attention mechanisms let a model weigh the relevance of different "
            "input tokens when producing each output token [1]. In transformers, "
            "this is computed via scaled dot-product attention across query, key, "
            "and value projections [2].",
            total_tokens=4310,
        )


class ScriptedOpenAIClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": ScriptedOpenAIChat()})()

    class embeddings:
        @staticmethod
        async def create(*, model, input):
            from openai.types import CreateEmbeddingResponse, Embedding
            from openai.types.create_embedding_response import Usage

            texts = input if isinstance(input, list) else [input]
            return CreateEmbeddingResponse(
                data=[Embedding(embedding=[0.1] * 1536, index=i, object="embedding") for i in range(len(texts))],
                model=model,
                object="list",
                usage=Usage(prompt_tokens=10, total_tokens=10),
            )


class FakePineconeIndex:
    def __init__(self):
        self.upserted_vectors: list = []

    async def upsert(self, **kwargs):
        self.upserted_vectors.extend(kwargs.get("vectors", []))

    async def query(self, **kwargs):
        from pinecone import QueryResponse, ScoredVector
        return QueryResponse(matches=[
            ScoredVector(
                id="doc-1-chunk-3",
                score=0.83,
                metadata={
                    "text": "Attention mechanisms compute a weighted sum over value vectors, "
                            "where weights come from a compatibility function between queries and keys.",
                    "source": "pinecone://ingested-paper-attention-is-all-you-need",
                    "title": "Attention Is All You Need (ingested)",
                },
            )
        ])


class FakePineconeVectorStore(pinecone_module.PineconeVectorStore):
    async def _get_index(self):
        return FakePineconeIndex()


class FakeTavilyClient:
    async def search(self, *, query, **kwargs):
        return {
            "results": [
                {
                    "content": f"A web article explaining: {query}",
                    "url": "https://example.com/attention-explained",
                    "title": "Understanding Attention",
                    "score": 0.79,
                }
            ]
        }


async def main() -> None:
    fake_openai = ScriptedOpenAIClient()
    fake_pinecone_store = FakePineconeVectorStore(index_host="fake", api_key="fake")
    fake_tavily = FakeTavilyClient()

    llm_client_module.get_client = lambda: fake_openai
    embeddings_module.get_client = lambda: fake_openai
    import app.graph.nodes as nodes_module
    nodes_module._vector_store = fake_pinecone_store
    tavily_module.get_tavily_client = lambda: fake_tavily

    initial_state: ResearchState = {
        "query": "explian how attension mechnisms work",
        "user_id": "demo-user-uuid",
        "run_id": str(uuid.uuid4()),
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
        "quota_ok": True,
    }

    print(f"\n{'=' * 70}")
    print("FULL-STACK DRY RUN — real orchestration code, mocked SDK boundaries")
    print(f"{'=' * 70}")
    print(f"Query (with deliberate typos): {initial_state['query']}\n")

    final_state = {}
    async for stream_mode, payload in research_graph.astream(
        initial_state, stream_mode=["updates", "values"]
    ):
        if stream_mode == "updates":
            for node_name, delta in payload.items():
                summary = {k: (v if not isinstance(v, str) or len(v) < 100 else v[:97] + "...") for k, v in delta.items()}
                print(f"  [{node_name}] -> {json.dumps(summary, default=str)[:160]}")
        else:
            final_state = payload

    print(f"\n{'-' * 70}")
    print(f"Corrected query (typos fixed by real decomposer prompt): {final_state['decomposition']['corrected_query']}")
    print(f"Chunks retrieved (vector + web, real orchestration code): {len(final_state['chunks'])}")
    print(f"Answer:\n{final_state['answer']}")
    print(f"Citations: {final_state['citations']}")
    print(f"OpenAI models called: {fake_openai.chat.completions.call_log}")
    print(f"Total tokens used this run: {final_state['tokens_used']}")

    from app.quota_store import quota_store

    usage = await quota_store.get_usage(initial_state["user_id"])
    print(f"Quota after this run: {usage.tokens_used}/{usage.quota_limit} tokens")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
