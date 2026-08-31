from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RunRecord:
    run_id: str
    user_id: str
    status: RunStatus = RunStatus.RUNNING
    events: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    result: dict | None = None
    created_at: float = field(default_factory=time.time)


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create_run(self, run_id: str, *, user_id: str) -> None:
        async with self._lock:
            self._runs[run_id] = RunRecord(run_id=run_id, user_id=user_id)

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def publish_event(self, run_id: str, event: dict) -> None:
        record = self._runs.get(run_id)
        if record is None:
            return
        record.events.append(event)
        for queue in record.subscribers:
            await queue.put(event)

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(run_id)

        queue: asyncio.Queue = asyncio.Queue()
        for past_event in record.events:
            queue.put_nowait(past_event)
        record.subscribers.append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        record = self._runs.get(run_id)
        if record is not None and queue in record.subscribers:
            record.subscribers.remove(queue)

    async def set_result(self, run_id: str, result: dict, *, status: RunStatus) -> None:
        record = self._runs.get(run_id)
        if record is not None:
            record.result = result
            record.status = status

    async def get_result(self, run_id: str) -> dict | None:
        record = self._runs.get(run_id)
        return record.result if record else None

    async def get_status(self, run_id: str) -> RunStatus | None:
        record = self._runs.get(run_id)
        return record.status if record else None

    async def owns(self, run_id: str, *, user_id: str) -> bool:
        record = self._runs.get(run_id)
        return record is not None and record.user_id == user_id


run_store = RunStore()


class SupabaseRunStore:
    def __init__(self, pool) -> None:
        self._pool = pool
        self._live = RunStore()

    async def create_run(self, run_id: str, *, user_id: str) -> None:
        await self._live.create_run(run_id, user_id=user_id)
        await self._pool.execute(
            "INSERT INTO research_runs (id, user_id, query, status) VALUES ($1, $2, '', 'running')",
            run_id,
            user_id,
        )

    async def publish_event(self, run_id: str, event: dict) -> None:
        await self._live.publish_event(run_id, event)

    async def subscribe(self, run_id: str):
        return await self._live.subscribe(run_id)

    def unsubscribe(self, run_id: str, queue) -> None:
        self._live.unsubscribe(run_id, queue)

    async def set_result(self, run_id: str, result: dict, *, status: RunStatus) -> None:
        await self._live.set_result(run_id, result, status=status)
        await self._pool.execute(
            """
            UPDATE research_runs
            SET status = $2, answer = $3, citations = $4::jsonb, cycle_count = $5, error = $6
            WHERE id = $1
            """,
            run_id,
            status.value,
            result.get("answer"),
            _json_or_none(result.get("citations")),
            result.get("cycle_count"),
            result.get("error"),
        )

    async def get_result(self, run_id: str) -> dict | None:
        row = await self._pool.fetchrow(
            "SELECT answer, citations, cycle_count, error FROM research_runs WHERE id = $1",
            run_id,
        )
        if row is None:
            return None
        return {
            "answer": row["answer"],
            "citations": _parse_json_or_none(row["citations"]),
            "cycle_count": row["cycle_count"],
            "error": row["error"],
        }

    async def get_status(self, run_id: str) -> RunStatus | None:
        row = await self._pool.fetchrow("SELECT status FROM research_runs WHERE id = $1", run_id)
        return RunStatus(row["status"]) if row else None

    async def owns(self, run_id: str, *, user_id: str) -> bool:
        row = await self._pool.fetchrow("SELECT user_id FROM research_runs WHERE id = $1", run_id)
        return row is not None and row["user_id"] == user_id


def _json_or_none(value):
    import json

    return json.dumps(value) if value is not None else None


def _parse_json_or_none(value):
    import json

    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


_supabase_run_store: SupabaseRunStore | None = None


def get_run_store():
    global _supabase_run_store
    from app.db import get_pool, is_db_enabled

    if not is_db_enabled():
        return run_store

    if _supabase_run_store is None:
        _supabase_run_store = SupabaseRunStore(get_pool())
    return _supabase_run_store
