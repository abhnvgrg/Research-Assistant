"""
In-memory run store — tracks status, streamed events, and final
results per run_id.

This is deliberately a thin, swappable seam: the real system stores
research_runs in Supabase (see the data-layer design) with RLS
scoping every row to auth.uid(). This in-memory version has the same
external shape (create/publish/subscribe/get_result) so swapping the
backing store later touches only this file, not the API routes or
the graph runner.

Concurrency note: asyncio.Queue is used per-subscriber so multiple
clients can independently stream the same run (e.g. a reconnecting
browser tab) — each gets a replay of past events followed by live
ones, rather than racing over a single shared queue.
"""

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
    events: list[dict] = field(default_factory=list)  # replay buffer
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
        """Appends to the replay buffer and fans out to every live
        subscriber's queue. A slow/disconnected subscriber never
        blocks others — each has its own queue."""
        record = self._runs.get(run_id)
        if record is None:
            return
        record.events.append(event)
        for queue in record.subscribers:
            await queue.put(event)

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        """Returns a queue pre-seeded with the full replay buffer so
        far, then wired to receive live events too — handles the
        case where a client connects to /stream slightly after the
        run started."""
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
        """async for interface parity with SupabaseRunStore, which
        genuinely needs a network round trip here — an in-memory dict
        lookup doesn't, but every caller must be able to `await`
        either backend identically without knowing which is active."""
        record = self._runs.get(run_id)
        return record.result if record else None

    async def get_status(self, run_id: str) -> RunStatus | None:
        record = self._runs.get(run_id)
        return record.status if record else None

    async def owns(self, run_id: str, *, user_id: str) -> bool:
        """RLS-equivalent check for the in-memory store — every
        lookup route must call this before returning any data, the
        same way Supabase RLS enforces `user_id = auth.uid()`."""
        record = self._runs.get(run_id)
        return record is not None and record.user_id == user_id


# Module-level singleton — a real deployment would replace this with
# a Supabase-backed implementation behind the same interface.
run_store = RunStore()


class SupabaseRunStore:
    """Postgres-backed implementation — but a genuine HYBRID, not a
    naive 1:1 swap, because live SSE fan-out (publish_event/subscribe/
    unsubscribe) is inherently process-local: Postgres has no push
    mechanism the app can `await` on directly, and the whole point of
    `asyncio.Queue` per subscriber is in-process delivery. Real-world
    architectures solve this with a message broker (Redis pub/sub —
    see the Docker/deployment design's Celery+Redis discussion) once
    you need cross-process fan-out; that's out of scope here.

    So: status/result/ownership are genuinely Postgres-backed (durable,
    correct across process restarts, matches SupabaseJobStore's
    approach exactly) — but publish_event/subscribe/unsubscribe
    delegate to an internal in-memory RunStore, exactly like the pure
    in-memory RunStore does, because there is no other correct choice
    without adding a message broker.

    Honest consequence, stated plainly rather than papered over: if
    the FastAPI process restarts mid-run, GET /research/{id}/result
    and /result-adjacent reads keep working immediately (Postgres is
    the source of truth) — but re-attaching a LIVE SSE stream to that
    run in the new process requires resume_graph_and_publish()
    (app/graph/checkpointing.py) to run first and re-populate an
    in-memory RunRecord, since subscribe() has nothing to attach to
    otherwise. This is the same boundary the checkpointer's own
    module docstring calls out for graph-level vs process-level
    resumability — the two gaps are the same gap, seen from two
    different files.
    """

    def __init__(self, pool) -> None:
        self._pool = pool
        self._live = RunStore()  # delegate for pub/sub only

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
        # Write through to both — Postgres for durability, the live
        # in-memory record too (harmless no-op if it doesn't exist,
        # e.g. after a restart with no resumed subscribers yet).
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
    """Returns the Supabase-backed hybrid store if app.db's
    connection pool initialized successfully, otherwise falls back to
    the pure in-memory RunStore. Callers (routes.py, sse.py,
    graph/runner.py) should call this instead of importing `run_store`
    directly."""
    global _supabase_run_store
    from app.db import get_pool, is_db_enabled

    if not is_db_enabled():
        return run_store

    if _supabase_run_store is None:
        _supabase_run_store = SupabaseRunStore(get_pool())
    return _supabase_run_store
