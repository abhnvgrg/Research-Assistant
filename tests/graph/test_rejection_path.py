from __future__ import annotations

import asyncio
import uuid

import app.graph.nodes as nodes
from app.graph.build import research_graph
from app.graph.state import ResearchState


async def fake_empty_topic_decomposer(query: str) -> dict:
    return {
        "topic_identified": False,
        "multi_intent": False,
        "recency_required": False,
        "corrected_query": query,
        "sub_questions": [],
    }


async def main() -> None:
    nodes._call_decomposer_llm = fake_empty_topic_decomposer

    initial_state: ResearchState = {
        "query": "what is the relationship between the thing and the other thing",
        "user_id": "demo-user-uuid",
        "run_id": str(uuid.uuid4()),
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
    }

    visited_nodes: list[str] = []
    final_state: dict = {}

    async for event in research_graph.astream(initial_state, stream_mode="updates"):
        for node_name in event:
            visited_nodes.append(node_name)
            final_state.update(event[node_name])

    print(f"Nodes visited: {visited_nodes}")
    print(f"Final answer: {final_state.get('answer')}")

    forbidden = {"vector_retriever", "web_search", "grader", "synthesizer", "reflection"}
    violated = forbidden & set(visited_nodes)

    assert "decomposer" in visited_nodes, "decomposer should always run"
    assert "error_handler" in visited_nodes, "topic_identified=False must route to error_handler"
    assert not violated, f"FAIL: retrieval/synthesis nodes ran despite no topic: {violated}"

    print("\nPASS: topic-less query correctly short-circuited before any retrieval or LLM synthesis call.")


if __name__ == "__main__":
    asyncio.run(main())
