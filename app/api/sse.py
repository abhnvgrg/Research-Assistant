from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import Request

from app.store import get_run_store

TERMINAL_EVENT_TYPES = {"run_complete", "error"}

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
                break

            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=DISCONNECT_POLL_INTERVAL_S
                )
            except asyncio.TimeoutError:
                continue

            yield format_sse(event)

            if event.get("type") in TERMINAL_EVENT_TYPES:
                break
    finally:
        store.unsubscribe(run_id, queue)
