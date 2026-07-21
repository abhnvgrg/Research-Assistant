"""
SSE formatting and the stream generator FastAPI's StreamingResponse
consumes.

Format matches what the Next.js useSSE() hook (from the frontend
streaming design) expects to parse: lines starting with "data:"
containing a JSON payload with a "type" field the frontend switches
on (node_start/node_complete/citation_map/run_complete/error).

Two important behaviors implemented here, both traced directly back
to earlier edge-case scrutiny:
  - Disconnect detection (edge case 6.2 — user closes tab mid-run):
    the generator checks request.is_disconnected() between events
    and stops cleanly rather than continuing to push events into a
    queue nobody is reading.
  - Terminal events (run_complete / error) end the stream themselves
    — the generator doesn't wait for the client to disconnect.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import Request

from app.store import get_run_store

TERMINAL_EVENT_TYPES = {"run_complete", "error"}

# How often to poll request.is_disconnected() while waiting for the
# next event — short enough that a closed tab is noticed quickly,
# long enough not to busy-loop.
DISCONNECT_POLL_INTERVAL_S = 0.5


def format_sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def event_stream_for_run(run_id: str, request: Request) -> AsyncGenerator[str, None]:
    store = get_run_store()

    try:
        queue = await store.subscribe(run_id)
    except KeyError:
        yield format_sse({"type": "error", "message": f"unknown run_id: {run_id}"})
        return

    try:
        while True:
            if await request.is_disconnected():
                break  # edge case 6.2 — stop pushing into a dead connection

            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=DISCONNECT_POLL_INTERVAL_S
                )
            except asyncio.TimeoutError:
                continue  # no event yet — loop back to the disconnect check

            yield format_sse(event)

            if event.get("type") in TERMINAL_EVENT_TYPES:
                break
    finally:
        store.unsubscribe(run_id, queue)
