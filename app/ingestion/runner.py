from __future__ import annotations

import logging

from app.graph.nodes import get_vector_store
from app.ingest_store import JobStatus, get_job_store
from app.ingestion.orchestrator import ingest_source

logger = logging.getLogger(__name__)


async def run_ingestion_and_publish(
    job_id: str,
    *,
    source_type: str,
    source: str | bytes,
    user_id: str,
    filename: str | None = None,
) -> None:
    store = get_job_store()
    try:
        result = await ingest_source(
            source_type=source_type,
            source=source,
            user_id=user_id,
            vector_store=get_vector_store(),
            filename=filename,
        )
        await store.set_result(job_id, result.to_dict(), status=JobStatus.COMPLETE)
    except Exception as exc:  # pragma: no cover - defensive runner guard
        logger.exception("Ingestion failed for job %s", job_id)
        await store.set_error(job_id, str(exc))

