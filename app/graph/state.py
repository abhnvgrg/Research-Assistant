"""
ResearchState — the shared state dict that flows through every node
in the LangGraph agent loop.

Reducer rules (this is the part everyone gets wrong):
  - Plain type annotation (str, int, dict)  -> OVERWRITE semantics.
    The last node to write this key wins. Use for values that should
    always reflect the most recent node's output.
  - Annotated[list, operator.add]           -> APPEND semantics.
    New values are concatenated onto the existing list. Use for
    anything that must accumulate across reflection cycles.

Getting this wrong is the single most common LangGraph bug: marking
`chunks` as a plain list means cycle 2's retrieval SILENTLY ERASES
cycle 1's chunks instead of adding to them. No exception is raised.
The system just quietly gets worse with more cycles instead of better.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class SubQuestion(TypedDict):
    """A single decomposed sub-question with routing metadata."""

    question: str
    intent: Literal["academic", "general", "recency"]
    aliases: list[str]


class Chunk(TypedDict):
    """A normalized retrieval result — same shape whether it came
    from Pinecone (vector) or Tavily (web), so the grader never needs
    to know the origin."""

    text: str
    source: str
    title: str
    score: float
    origin: Literal["vector", "web"]


class GradedChunk(Chunk):
    """A chunk after the relevance grader has scored it."""

    relevant: bool
    grade_score: float
    grade_reason: str


class ReflectionResult(TypedDict):
    """Output of the reflection node's self-critique."""

    pass_: bool  # trailing underscore — 'pass' is a Python keyword
    checklist: dict[str, bool | str]
    gap: str


class DecomposerOutput(TypedDict):
    """Raw structured output from the query decomposer node."""

    topic_identified: bool
    multi_intent: bool
    recency_required: bool
    corrected_query: str
    sub_questions: list[SubQuestion]


class ResearchState(TypedDict):
    # ---- input (set once, never mutated) ----
    query: str
    user_id: str
    run_id: str

    # ---- decomposition (overwrite — replaced once per cycle) ----
    decomposition: NotRequired[DecomposerOutput]

    # ---- routing ----
    route: NotRequired[Literal["vector", "web", "both"]]

    # ---- retrieval (APPEND — must accumulate across reflection cycles) ----
    chunks: Annotated[list[Chunk], operator.add]

    # ---- grading (overwrite — fully recomputed each grading pass) ----
    graded: NotRequired[list[GradedChunk]]

    # ---- synthesis (overwrite — last synthesis is the current answer) ----
    answer: NotRequired[str]
    citations: NotRequired[dict[str, str]]  # {"1": url, "2": url, ...}

    # ---- reflection (APPEND — keep every past gap for the Router to use) ----
    reflection_history: Annotated[list[ReflectionResult], operator.add]
    cycle_count: int  # overwrite — incremented manually by the router node

    # ---- control / error handling ----
    quota_ok: NotRequired[bool]
    degraded: NotRequired[bool]  # True if Pinecone failed and we're web-only
    error: NotRequired[str | None]

    # ---- observability ----
    # APPEND (via operator.add on ints = sum) — every node that makes
    # a real LLM call reports its own token delta here, so the total
    # across a run (including every reflection cycle's re-grading and
    # re-synthesis) accumulates correctly. This is what
    # quota_store.add_usage() records after a run completes — see
    # formatter_node. Getting this wrong the overwrite way (plain
    # `int`, not Annotated[int, operator.add]) would silently under-
    # report cost on any run with more than one reflection cycle,
    # exactly the same class of bug we caught with the `chunks` field
    # earlier in this project.
    tokens_used: Annotated[int, operator.add]
    mlflow_run_id: NotRequired[str]
