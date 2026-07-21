"""
Node functions for the research assistant agent graph.

Every node follows the same contract:
  - Receives the full ResearchState
  - Returns a PARTIAL dict — only the keys it's updating
  - Never raises uncaught — wraps risky calls in try/except and
    writes to state["error"] instead, so a bad LLM call degrades
    the run gracefully rather than killing the SSE stream outright.

External calls (OpenAI, Pinecone, Tavily) are stubbed behind small
async functions marked with `# REAL:` comments showing exactly what
to swap in. This keeps the graph runnable and testable today, and
makes the integration points unambiguous later.
"""

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

# Module-level singleton — lazily constructs its Pinecone client on
# first use (see PineconeVectorStore._get_index), so importing this
# module never requires PINECONE_API_KEY to be set.
_vector_store = PineconeVectorStore()


def get_vector_store() -> PineconeVectorStore:
    """Public accessor for the shared vector store singleton — used
    by the ingestion pipeline so query-time retrieval and offline
    ingestion write to the exact same Pinecone index without each
    constructing their own client. Reads the module global at CALL
    time (not import time), so existing tests that monkeypatch
    nodes._vector_store directly continue to work unchanged."""
    return _vector_store

MAX_CYCLES = 3
GRADE_THRESHOLD = 0.5
MAX_CHUNKS_TO_SYNTH = 7

# Node-level model selection (from our cost modeling design):
# gpt-4o-mini for classification-style tasks (decomposition, grading,
# reflection checklist evaluation), gpt-4o only for long-form
# synthesis where quality genuinely justifies the ~25x cost delta.
MODEL_DECOMPOSER = "gpt-4o-mini"
MODEL_GRADER = "gpt-4o-mini"
MODEL_SYNTHESIZER = "gpt-4o"
MODEL_REFLECTION = "gpt-4o-mini"


# --------------------------------------------------------------------------
# Stubbed external calls — replace these bodies with real API calls.
# Node functions below never call OpenAI/Pinecone/Tavily directly;
# they only call these, so swapping in real APIs touches one place.
# --------------------------------------------------------------------------

async def _call_decomposer_llm(query: str) -> tuple[dict, int]:
    """Real gpt-4o-mini call. Raises LLMCallError on failure — caught
    by decomposer_node and written to state['error']. Returns
    (decomposition, tokens_used) — see app/llm/client.py's
    call_json() for where tokens_used comes from."""
    return await call_json(
        model=MODEL_DECOMPOSER,
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        user_prompt=f"QUERY: {query}",
    )


async def _call_vector_retriever(sub_questions: list[dict], user_id: str) -> list[Chunk]:
    """Embeds each sub-question and queries Pinecone in the user's
    namespace (derived upstream from the verified JWT sub — never
    from client input, per our security design). Fires one query
    per sub-question concurrently; a real production version would
    also batch the embedding calls via embed_texts() for fewer
    OpenAI round-trips, left as a follow-up optimization."""
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
    """One Tavily call per sub-question, fired concurrently. search_
    depth='advanced' (set in search_web) is what makes Tavily return
    full extracted article text rather than short snippets."""
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
    """Real gpt-4o-mini call, one chunk per call — fired concurrently
    via asyncio.gather from grader_node below. Chunk text is
    truncated to 500 chars (per design) and never batched with other
    chunks in one prompt, which avoids LLM position bias. Returns
    (grade, tokens_used) per chunk — grader_node sums these across
    every chunk graded in a pass."""
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
    """Post-generation validation (edge case 4.1 — citation
    hallucination). Finds every [N] in the answer text, strips any
    reference to an N that doesn't correspond to a real graded
    chunk, and builds the citation map for the frontend."""
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
    """Real gpt-4o call. Chunks are wrapped in <chunk> XML tags with
    the source and index as attributes — this is the structural
    prompt-injection defense from our security design: the system
    prompt instructs the model to treat this content as data, never
    as instructions, regardless of what's inside it. Returns a dict
    (not a tuple, unlike the other three _call_*_llm functions) since
    synthesizer_node already needs a dict for answer/citations —
    tokens_used is folded into that same dict for consistency."""
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
    """Real gpt-4o-mini checklist evaluation against Q1-Q4. Returns
    (reflection_result, tokens_used)."""
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


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------

async def quota_check_node(state: ResearchState) -> dict:
    """Guard node — checked before any LLM call is made. Real check
    against QuotaStore (an in-memory stand-in for the Supabase
    users.tokens_used/quota_limit columns — see app/quota_store.py),
    or the real Supabase-backed store once app.db's connection pool
    is available — get_quota_store() picks whichever is active. A
    user already at or over their limit is blocked from starting a
    new run entirely; there is no partial-credit path where a run
    starts and gets cut off mid-way, since we can't know in advance
    how many tokens a given query will actually consume."""
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
    """Decides vector vs web vs both, and increments cycle_count.
    On reflection cycles >1, this is also where the Router would
    rewrite sub-questions using reflection_history[-1]['gap'] —
    stubbed here as a pass-through."""
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
        # Degraded mode — Pinecone failed, don't kill the run, just
        # flag it and continue with whatever web_search returns.
        return {"degraded": True, "error": f"vector_retrieval_failed: {e}"}


async def web_search_node(state: ResearchState) -> dict:
    try:
        sub_qs = state["decomposition"]["sub_questions"]
        chunks = await _call_web_search(sub_qs)
        return {"chunks": chunks}
    except Exception as e:
        return {"error": f"web_search_failed: {e}"}


async def grader_node(state: ResearchState) -> dict:
    """Grades every chunk in PARALLEL via asyncio.gather — never
    batch multiple chunks into one LLM call (position bias)."""
    try:
        query = state["decomposition"]["corrected_query"]
        chunks = state["chunks"]

        graded_raw = await asyncio.gather(
            *[_call_grader_llm(c, query) for c in chunks]
        )
        # Each element is (grade_dict, tokens_used) — sum tokens
        # across every chunk graded in this pass.
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
    """Guards against the most dangerous silent failure in the whole
    system: calling the LLM with zero graded chunks, which produces
    a confident, uncited, hallucinated answer that looks identical
    to a real one."""
    graded = state.get("graded", [])

    if not graded:
        return {
            "answer": (
                "I wasn't able to find sufficient sources to answer this "
                "confidently. Try rephrasing the question or narrowing its scope."
            ),
            "citations": {},
            "tokens_used": 0,  # no LLM call made — nothing to charge for
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
            "reflection_history": [result],  # appended via operator.add
            "tokens_used": tokens,
        }
    except Exception as e:
        # Fail open: if reflection itself breaks, don't loop forever —
        # treat it as a pass so the run still completes. No tokens
        # charged since we don't know how much of the call, if any,
        # actually completed before the failure.
        return {
            "reflection_history": [
                {"pass_": True, "checklist": {}, "gap": f"reflection_error: {e}"}
            ],
            "tokens_used": 0,
        }


async def formatter_node(state: ResearchState) -> dict:
    """Final node — builds the response payload AND records real
    token usage against the user's quota. This is what makes
    quota_check_node meaningful: without recording usage somewhere,
    'checking quota' would just be checking a number that never
    changes — exactly as fake as the old always-True stub. REAL: this
    is also where mlflow.log_params/log_metrics/log_artifact calls
    would happen, each wrapped in its own try/except so observability
    code can never break the critical path — not yet implemented."""
    mlflow_run_id = str(uuid.uuid4())
    await _record_usage(state)
    return {"mlflow_run_id": mlflow_run_id}


async def _record_usage(state: ResearchState) -> None:
    """Shared by formatter_node and error_handler_node — every real
    LLM call that ran before a failure still spent real money and
    must still count against the user's quota, even if the run never
    reached a successful answer. Recording usage must never itself
    raise and break the response path; worst case on failure is
    under-billed usage, not a broken user-facing result."""
    try:
        store = get_quota_store()
        await store.add_usage(state["user_id"], state.get("tokens_used", 0))
    except Exception:  # pragma: no cover - defensive
        pass


async def error_handler_node(state: ResearchState) -> dict:
    """Terminal node for any branch that set state['error']. In the
    real FastAPI integration this is where an SSE 'error' event gets
    emitted before the stream closes cleanly. Also records usage —
    see _record_usage() — since a rejected query (e.g. topic-less,
    per check_decomposition) still made one real, billable decomposer
    call before being rejected."""
    await _record_usage(state)
    return {
        "answer": f"Run failed: {state.get('error', 'unknown error')}",
        "citations": {},
    }
