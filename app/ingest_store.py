from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    user_id: str
    status: JobStatus = JobStatus.PROCESSING
    result: dict | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    async def create_job(self, job_id: str, *, user_id: str) -> None:
        self._jobs[job_id] = JobRecord(job_id=job_id, user_id=user_id)

    async def set_result(self, job_id: str, result: dict, *, status: JobStatus) -> None:
        record = self._jobs.get(job_id)
        if record is not None:
            record.result = result
            record.status = status

    async def set_error(self, job_id: str, error: str) -> None:
        record = self._jobs.get(job_id)
        if record is not None:
            record.error = error
            record.status = JobStatus.FAILED

    async def get_result(self, job_id: str) -> dict | None:
        record = self._jobs.get(job_id)
        return record.result if record else None

    async def get_status(self, job_id: str) -> JobStatus | None:
        record = self._jobs.get(job_id)
        return record.status if record else None

    async def get_error(self, job_id: str) -> str | None:
        record = self._jobs.get(job_id)
        return record.error if record else None

    async def owns(self, job_id: str, *, user_id: str) -> bool:
        record = self._jobs.get(job_id)
        return record is not None and record.user_id == user_id


class SupabaseJobStore:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def create_job(self, job_id: str, *, user_id: str) -> None:
        await self._pool.execute(
            "INSERT INTO documents (id, user_id, status) VALUES ($1, $2, 'processing')",
            job_id,
            user_id,
        )

    async def set_result(self, job_id: str, result: dict, *, status: JobStatus) -> None:
        await self._pool.execute(
            """
            UPDATE documents
            SET status = $2, title = $3, source = $4, chunks_ingested = $5, chunks_dropped = $6, error = NULL
            WHERE id = $1
            """,
            job_id,
            status.value,
            result.get("title"),
            result.get("source"),
            result.get("chunks_ingested"),
            result.get("chunks_dropped"),
        )

    async def set_error(self, job_id: str, error: str) -> None:
        await self._pool.execute(
            "UPDATE documents SET status = 'failed', error = $2 WHERE id = $1",
            job_id,
            error,
        )

    async def get_result(self, job_id: str) -> dict | None:
        row = await self._pool.fetchrow(
            "SELECT title, source, chunks_ingested, chunks_dropped FROM documents WHERE id = $1",
            job_id,
        )
        if row is None:
            return None
        return {
            "title": row["title"],
            "source": row["source"],
            "chunks_ingested": row["chunks_ingested"],
            "chunks_dropped": row["chunks_dropped"],
        }

    async def get_status(self, job_id: str) -> JobStatus | None:
        row = await self._pool.fetchrow("SELECT status FROM documents WHERE id = $1", job_id)
        return JobStatus(row["status"]) if row else None

    async def get_error(self, job_id: str) -> str | None:
        row = await self._pool.fetchrow("SELECT error FROM documents WHERE id = $1", job_id)
        return row["error"] if row else None

    async def owns(self, job_id: str, *, user_id: str) -> bool:
        row = await self._pool.fetchrow("SELECT user_id FROM documents WHERE id = $1", job_id)
        return row is not None and row["user_id"] == user_id


job_store = JobStore()
_supabase_job_store: SupabaseJobStore | None = None


def get_job_store():
    global _supabase_job_store
    from app.db import get_pool, is_db_enabled

    if not is_db_enabled():
        return job_store

    if _supabase_job_store is None:
        _supabase_job_store = SupabaseJobStore(get_pool())
    return _supabase_job_store

