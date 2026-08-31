from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class SubQuestion(TypedDict):
    question: str
    intent: Literal["academic", "general", "recency"]
    aliases: list[str]


class Chunk(TypedDict):
    text: str
    source: str
    title: str
    score: float
    origin: Literal["vector", "web"]


class GradedChunk(Chunk):
    relevant: bool
    grade_score: float
    grade_reason: str


class ReflectionResult(TypedDict):
    pass_: bool
    checklist: dict[str, bool | str]
    gap: str


class DecomposerOutput(TypedDict):
    topic_identified: bool
    multi_intent: bool
    recency_required: bool
    corrected_query: str
    sub_questions: list[SubQuestion]


class ResearchState(TypedDict):
    query: str
    user_id: str
    run_id: str

    decomposition: NotRequired[DecomposerOutput]

    route: NotRequired[Literal["vector", "web", "both"]]

    chunks: Annotated[list[Chunk], operator.add]

    graded: NotRequired[list[GradedChunk]]

    answer: NotRequired[str]
    citations: NotRequired[dict[str, str]]

    reflection_history: Annotated[list[ReflectionResult], operator.add]
    cycle_count: int

    quota_ok: NotRequired[bool]
    degraded: NotRequired[bool]
    error: NotRequired[str | None]

    tokens_used: Annotated[int, operator.add]
    mlflow_run_id: NotRequired[str]
