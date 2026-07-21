"""
Graph runner — the background task that actually executes the
research graph and publishes streamed events to the RunStore.

Key design point: uses stream_mode=["updates", "values"] in a SINGLE
astream() call. 'updates' chunks ({node_name: delta}) become live
node_start/node_complete-shaped SSE events; the LAST 'values' chunk
(the fully-reduced state after the final superstep) becomes the
final result. This avoids the state-reconstruction bug we caught
earlier in run_demo.py (naively dict.update()-ing deltas
under-reports accumulated fields) without needing to run the graph
twice.

Every exception is caught here — this is the node-level "never let
an exception kill the SSE stream" principle applied at the top level:
if the graph itself raises somehow, the run is marked failed and an
error event is published, rather than leaving the run stuck in
RUNNING forever with subscribers hanging on an empty queue.

Every astream() call passes config={"configurable": {"thread_id":
run_id}} — this is what ties LangGraph's checkpoints (when a
checkpointer is configured; see app/graph/checkpointing.py) to a
specific run. It's harmless when no checkpointer is attached (the
default for tests and demo scripts), so this is always safe to pass.
"""

from __future__ import annotations

import logging

from app.graph.checkpointing import get_research_graph
from app.graph.state import ResearchState
from app.store import RunStatus, get_run_store

logger = logging.getLogger(__name__)


async def run_graph_and_publish(
    run_id: str, initial_state: ResearchState | None
) -> None:
    """initial_state=None means RESUME from the last checkpoint for
    this run_id instead of starting fresh — only meaningful when a
    real checkpointer is attached (get_research_graph() returns the
    checkpointed graph) and a checkpoint genuinely exists for this
    thread_id. Passing None with no checkpointer, or no existing
    checkpoint, raises inside astream() and is caught by the
    top-level exception guard below like any other failure."""
    graph = get_research_graph()
    run_store = get_run_store()
    config = {"configurable": {"thread_id": run_id}}
    final_state: dict = {}

    try:
        async for mode, chunk in graph.astream(
            initial_state, stream_mode=["updates", "values"], config=config
        ):
            if mode == "updates":
                for node_name, delta in chunk.items():
                    await run_store.publish_event(
                        run_id,
                        {
                            "type": "node_complete",
                            "node": node_name,
                            # Only forward small, JSON-safe fields to
                            # the client — never leak full chunk text
                            # or internal state shape over SSE.
                            "summary": _summarize_delta(node_name, delta),
                        },
                    )
            elif mode == "values":
                final_state = chunk  # last one wins == true final state

    except Exception as e:  # pragma: no cover - defensive top-level guard
        logger.exception("Graph execution failed for run %s", run_id)
        await run_store.publish_event(run_id, {"type": "error", "message": str(e)})
        await run_store.set_result(run_id, {"error": str(e)}, status=RunStatus.FAILED)
        return

    if final_state.get("error"):
        await run_store.publish_event(
            run_id, {"type": "error", "message": final_state["error"]}
        )
        await run_store.set_result(run_id, final_state, status=RunStatus.FAILED)
        return

    await run_store.publish_event(
        run_id,
        {
            "type": "citation_map",
            "citations": final_state.get("citations", {}),
        },
    )
    await run_store.publish_event(
        run_id,
        {
            "type": "run_complete",
            "answer": final_state.get("answer", ""),
            "cycle_count": final_state.get("cycle_count", 0),
        },
    )
    await run_store.set_result(run_id, final_state, status=RunStatus.COMPLETE)


async def resume_graph_and_publish(run_id: str) -> None:
    """Resumes a run from its last checkpoint instead of starting
    fresh — the actual resumability feature this module exists for.
    Requires a real checkpointer to be attached AND a checkpoint to
    already exist for run_id (i.e. the run got at least one
    superstep in before whatever interrupted it). Now that
    get_run_store() can return a Supabase-backed store, the run
    record itself also survives a process restart — the remaining
    gap is re-populating an in-memory RunRecord for live SSE
    subscribers to attach to, which this function does simply by
    being called at all (run_graph_and_publish publishes events
    through whichever store get_run_store() currently returns)."""
    await run_graph_and_publish(run_id, None)


def _summarize_delta(node_name: str, delta: dict) -> dict:
    """Builds a small, client-safe summary per node — deliberately
    NOT the raw delta, since that could include full chunk text,
    internal error strings, etc. that shouldn't cross the wire
    verbatim. Extend this per node as the frontend needs more detail."""
    if node_name == "decomposer" and "decomposition" in delta:
        decomp = delta["decomposition"]
        return {
            "corrected_query": decomp.get("corrected_query"),
            "sub_question_count": len(decomp.get("sub_questions", [])),
        }
    if node_name == "grader" and "graded" in delta:
        return {"chunks_passed_grading": len(delta["graded"])}
    if node_name in ("vector_retriever", "web_search") and "chunks" in delta:
        return {"chunks_retrieved": len(delta["chunks"])}
    if node_name == "reflection" and "reflection_history" in delta:
        latest = delta["reflection_history"][-1] if delta["reflection_history"] else {}
        return {"passed": latest.get("pass_"), "gap": latest.get("gap")}
    return {}
