from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=10,
        max_length=1000,
        description="The research question. For longer content, use /ingest/document instead.",
    )


class QueryResponse(BaseModel):
    run_id: str
    status: str


class ResultResponse(BaseModel):
    run_id: str
    status: str
    answer: str | None = None
    citations: dict[str, str] | None = None
    cycle_count: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    job_id: str
    status: str


class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    title: str | None = None
    source: str | None = None
    chunks_ingested: int | None = None
    chunks_dropped: int | None = None
    error: str | None = None
