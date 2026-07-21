"""
Manual smoke test for the research graph — run directly to see the
full agent loop execute with stubbed external calls, and to print
every node's state delta as it streams.

Run with:  python -m app.graph.run_demo
"""

from __future__ import annotations

import asyncio
import json
import uuid

from app.graph.build import research_graph
from app.graph.state import ResearchState


async def run_query(query: str) -> None:
    initial_state: ResearchState = {
        "query": query,
        "user_id": "demo-user-uuid",
        "run_id": str(uuid.uuid4()),
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}")

    final_state: dict = {}

    # stream_mode="updates" emits {node_name: state_delta} after each
    # superstep — this is what powers the real SSE node_start/
    # node_complete events in the FastAPI integration. We print from
    # this stream, but we must NOT reconstruct final state by naively
    # dict.update()-ing deltas — that ignores reducer semantics
    # (operator.add) and silently under-counts accumulated lists.
    # Instead we re-run with stream_mode="values" to get the true,
    # fully-reduced state after the final superstep.
    async for event in research_graph.astream(initial_state, stream_mode="updates"):
        for node_name, delta in event.items():
            printable = {
                k: (v if not isinstance(v, str) or len(v) < 80 else v[:77] + "...")
                for k, v in delta.items()
            }
            print(f"\n  [{node_name}]")
            print(f"    -> {json.dumps(printable, default=str, indent=6)[1:-1].strip()}")

    # Reset the stub's internal call counter so the second pass
    # (stream_mode="values") sees the exact same reflection behavior
    # as the first pass (stream_mode="updates") — otherwise the two
    # runs would diverge after call 3 since the stub is stateful.
    from app.graph.nodes import _reflection_call_count
    _reflection_call_count["n"] = 0

    # stream_mode="values" yields the full ResearchState after every
    # superstep — the last one emitted is the true final reduced state.
    async for snapshot in research_graph.astream(initial_state, stream_mode="values"):
        final_state = snapshot

    print(f"\n{'-' * 70}")
    print(f"FINAL ANSWER: {final_state.get('answer')}")
    print(f"CITATIONS: {final_state.get('citations')}")
    print(f"CYCLES USED: {final_state.get('cycle_count')}")
    print(f"DEGRADED: {final_state.get('degraded', False)}")
    print(f"ERROR: {final_state.get('error')}")
    print(f"TOTAL CHUNKS ACCUMULATED (proves operator.add reducer): {len(final_state.get('chunks', []))}")
    print(f"REFLECTION HISTORY LENGTH (proves it appended, not overwrote): {len(final_state.get('reflection_history', []))}")
    print(f"{'=' * 70}\n")


async def main() -> None:
    await run_query("explain how transformers work")


if __name__ == "__main__":
    asyncio.run(main())
