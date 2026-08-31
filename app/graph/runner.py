from __future__ import annotations

import logging

from app.graph.checkpointing import get_research_graph
from app.graph.state import ResearchState
from app.store import RunStatus, get_run_store

logger = logging.getLogger(__name__)


async def run_graph_and_publish(
    run_id: str, initial_state: ResearchState | None
) -> None:
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
                            "summary": _summarize_delta(node_name, delta),
                        },
                    )
            elif mode == "values":
                final_state = chunk

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
    await run_graph_and_publish(run_id, None)


def _summarize_delta(node_name: str, delta: dict) -> dict:
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
