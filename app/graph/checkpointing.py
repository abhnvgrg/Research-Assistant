from __future__ import annotations

import logging
import os
import tempfile
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.build import build_research_graph, research_graph

logger = logging.getLogger(__name__)

_checkpointer: AsyncSqliteSaver | None = None
_checkpointer_cm: AbstractAsyncContextManager[AsyncSqliteSaver] | None = None
_checkpointed_graph = None


def _checkpoint_db_path() -> str:
    return os.environ.get(
        "CHECKPOINT_DB_PATH",
        os.path.join(tempfile.gettempdir(), "research_assistant_checkpoints.db"),
    )


async def init_checkpointer() -> None:
    """Best-effort initialization.

    On any failure, keep serving requests using the non-checkpointed graph.
    """
    global _checkpointer, _checkpointer_cm, _checkpointed_graph

    if _checkpointer is not None:
        return

    try:
        db_path = _checkpoint_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = db_path.replace("\\", "/")
        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(conn)
        _checkpointer = await _checkpointer_cm.__aenter__()
        _checkpointed_graph = build_research_graph(checkpointer=_checkpointer)
    except Exception as exc:  # pragma: no cover - defensive startup fallback
        logger.warning("Checkpoint initialization failed; continuing without it: %s", exc)
        _checkpointer = None
        _checkpointed_graph = None
        _checkpointer_cm = None


async def close_checkpointer() -> None:
    global _checkpointer, _checkpointer_cm, _checkpointed_graph

    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)

    _checkpointer = None
    _checkpointer_cm = None
    _checkpointed_graph = None


def is_checkpointing_enabled() -> bool:
    return _checkpointed_graph is not None


def get_research_graph():
    return _checkpointed_graph or research_graph

