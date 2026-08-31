from __future__ import annotations

import asyncio
import uuid

from app.graph.state import Chunk, GradedChunk, ResearchState
from app.llm.client import LLMCallError, call_json, call_text
from app.llm.embeddings import embed_text
from app.quota_store import get_quota_store
from app.llm.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from app.retrieval.pinecone_store import PineconeVectorStore
from app.retrieval.tavily_search import search_web

_vector_store = PineconeVectorStore()


def get_vector_store() -> PineconeVectorStore:
    return _vector_store

MAX_CYCLES = 3
GRADE_THRESHOLD = 0.5
MAX_CHUNKS_TO_SYNTH = 7

MODEL_DECOMPOSER = "gpt-4o-mini"
MODEL_GRADER = "gpt-4o-mini"
MODEL_SYNTHESIZER = "gpt-4o"
MODEL_REFLECTION = "gpt-4o-mini"


async def _call_decomposer_llm(query: str) -> tuple[dict, int]:
    return await call_json(
        model=MODEL_DECOMPOSER,
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        user_prompt=f"QUERY: {query}",
    )


async def _call_vector_retriever(sub_questions: list[dict], user_id: str) -> list[Chunk]:
    async def query_one(sub_q: dict) -> list[Chunk]:
        vector = await embed_text(sub_q["question"])
        matches = await _vector_store.query(vector=vector, top_k=12, namespace=user_id)
        return [
            {
                "text": m["text"],
                "source": m["source"],
                "title": m["title"],
                "score": m["score"],
                "origin": "vector",
            }
            for m in matches
        ]

    results = await asyncio.gather(*[query_one(sq) for sq in sub_questions])
    return [chunk for sublist in results for chunk in sublist]


async def _call_web_search(sub_questions: list[dict]) -> list[Chunk]:
    async def search_one(sub_q: dict) -> list[Chunk]:
        results = await search_web(sub_q["question"], max_results=5)
        return [
            {
                "text": r["text"],
                "source": r["source"],
                "title": r["title"],
                "score": r["score"],
                "origin": "web",
            }
            for r in results
        ]

    results = await asyncio.gather(*[search_one(sq) for sq in sub_questions])
    return [chunk for sublist in results for chunk in sublist]


async def _call_grader_llm(chunk: Chunk, query: str) -> tuple[dict, int]:
    user_prompt = (
        f"QUESTION: {query}\n"
        f"CHUNK: {chunk['text'][:500]}\n"
        f"SOURCE: {chunk['source']}"
    )
    return await call_json(
        model=MODEL_GRADER,
        system_prompt=GRADER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def _extract_and_validate_citations(answer: str, graded: list[GradedChunk]) -> tuple[str, dict]:
    import re

    valid_indices = set(range(1, len(graded) + 1))
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    invalid = found - valid_indices

    cleaned_answer = answer
    for bad_index in invalid:
        cleaned_answer = cleaned_answer.replace(f"[{bad_index}]", "")

    citations = {
        str(i + 1): chunk["source"]
        for i, chunk in enumerate(graded)
        if (i + 1) in found and (i + 1) in valid_indices
    }
    return cleaned_answer, citations


async def _call_synthesizer_llm(query: str, graded: list[GradedChunk]) -> dict:
    chunk_blocks = "\n".join(
        f'<chunk index="{i + 1}" source="{c["source"]}">{c["text"]}</chunk>'
        for i, c in enumerate(graded)
    )
    user_prompt = f"ORIGINAL QUERY: {query}\n\nSOURCES:\n{chunk_blocks}"

    raw_answer, tokens_used = await call_text(
        model=MODEL_SYNTHESIZER,
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    cleaned_answer, citations = _extract_and_validate_citations(raw_answer, graded)
    return {"answer": cleaned_answer, "citations": citations, "tokens_used": tokens_used}


async def _call_reflection_llm(query: str, sub_questions: list[dict], answer: str) -> tuple[dict, int]:
    sub_q_text = "\n".join(f"- {sq['question']}" for sq in sub_questions)
    user_prompt = (
        f"ORIGINAL QUERY: {query}\n"
        f"SUB-QUESTIONS:\n{sub_q_text}\n\n"
        f"SYNTHESIZED ANSWER:\n{answer}"
    )
    return await call_json(
        model=MODEL_REFLECTION,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


async def quota_check_node(state: ResearchState) -> dict:
    try:
        store = get_quota_store()
        allowed = await store.has_quota(state["user_id"])
        if not allowed:
            usage = await store.get_usage(state["user_id"])
            return {
                "quota_ok": False,
                "error": (
                    f"quota_exceeded: {usage.tokens_used}/{usage.quota_limit} "
                    "tokens used this billing period"
                ),
            }
        return {"quota_ok": True, "error": None}
    except Exception as e:  # pragma: no cover - defensive
        return {"quota_ok": False, "error": f"quota_check_failed: {e}"}


async def decomposer_node(state: ResearchState) -> dict:
    try:
        result, tokens = await _call_decomposer_llm(state["query"])
        return {"decomposition": result, "tokens_used": tokens, "error": None}
    except Exception as e:
        return {"tokens_used": 0, "error": f"decomposition_failed: {e}"}


async def router_node(state: ResearchState) -> dict:
    decomp = state["decomposition"]
    recency = decomp.get("recency_required", False)

    if recency:
        route = "web"
    else:
        route = "both"

    return {
        "route": route,
        "cycle_count": state.get("cycle_count", 0) + 1,
    }


async def vector_retriever_node(state: ResearchState) -> dict:
    try:
        sub_qs = state["decomposition"]["sub_questions"]
        chunks = await _call_vector_retriever(sub_qs, state["user_id"])
        return {"chunks": chunks}
    except Exception as e:
        return {"degraded": True, "error": f"vector_retrieval_failed: {e}"}


async def web_search_node(state: ResearchState) -> dict:
    try:
        sub_qs = state["decomposition"]["sub_questions"]
        chunks = await _call_web_search(sub_qs)
        return {"chunks": chunks}
    except Exception as e:
        return {"error": f"web_search_failed: {e}"}


async def grader_node(state: ResearchState) -> dict:
    try:
        query = state["decomposition"]["corrected_query"]
        chunks = state["chunks"]

        graded_raw = await asyncio.gather(
            *[_call_grader_llm(c, query) for c in chunks]
        )
        total_tokens = sum(tokens for _grade, tokens in graded_raw)

        graded: list[GradedChunk] = [
            {
                **chunk,
                "relevant": g["relevant"],
                "grade_score": g["score"],
                "grade_reason": g["reason"],
            }
            for chunk, (g, _tokens) in zip(chunks, graded_raw)
        ]

        passed = [g for g in graded if g["grade_score"] >= GRADE_THRESHOLD]
        passed.sort(key=lambda g: g["grade_score"], reverse=True)

        return {"graded": passed[:MAX_CHUNKS_TO_SYNTH], "tokens_used": total_tokens}
    except Exception as e:
        return {"graded": [], "tokens_used": 0, "error": f"grading_failed: {e}"}


async def synthesizer_node(state: ResearchState) -> dict:
    graded = state.get("graded", [])

    if not graded:
        return {
            "answer": (
                "I wasn't able to find sufficient sources to answer this "
                "confidently. Try rephrasing the question or narrowing its scope."
            ),
            "citations": {},
            "tokens_used": 0,
        }

    try:
        result = await _call_synthesizer_llm(state["query"], graded)
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "tokens_used": result["tokens_used"],
            "error": None,
        }
    except Exception as e:
        return {"tokens_used": 0, "error": f"synthesis_failed: {e}"}


async def reflection_node(state: ResearchState) -> dict:
    try:
        result, tokens = await _call_reflection_llm(
            state["query"],
            state["decomposition"]["sub_questions"],
            state.get("answer", ""),
        )
        return {
            "reflection_history": [result],
            "tokens_used": tokens,
        }
    except Exception as e:
        return {
            "reflection_history": [
                {"pass_": True, "checklist": {}, "gap": f"reflection_error: {e}"}
            ],
            "tokens_used": 0,
        }


async def formatter_node(state: ResearchState) -> dict:
    mlflow_run_id = str(uuid.uuid4())
    await _record_usage(state)
    return {"mlflow_run_id": mlflow_run_id}


async def _record_usage(state: ResearchState) -> None:
    try:
        store = get_quota_store()
        await store.add_usage(state["user_id"], state.get("tokens_used", 0))
    except Exception:  # pragma: no cover - defensive
        pass


async def error_handler_node(state: ResearchState) -> dict:
    await _record_usage(state)
    return {
        "answer": f"Run failed: {state.get('error', 'unknown error')}",
        "citations": {},
    }
