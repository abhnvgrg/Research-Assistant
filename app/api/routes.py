"""
API routes — implements the endpoint surface from the API design
session: POST /research/query returns a run_id immediately (per the
"streaming endpoint is critical for UX" design — the client opens
the SSE connection right after, without waiting for any work to
happen synchronously), GET /research/{run_id}/stream is the SSE
endpoint, GET /research/{run_id}/result is the polling fallback /
post-completion fetch.

Every route that touches a specific run_id checks run_store.owns()
first — this is the in-memory equivalent of Supabase RLS
(`user_id = auth.uid()`), the same defense-in-depth principle from
the auth design: even if something upstream got the user_id wrong,
this check independently blocks cross-user access to another user's
run (edge case: IDOR / insecure direct object reference).
"""

from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user_id
from app.api.schemas import (
    HealthResponse,
    IngestResponse,
    IngestStatusResponse,
    QueryRequest,
    QueryResponse,
    ResultResponse,
)
from app.api.sse import event_stream_for_run
from app.graph.runner import run_graph_and_publish
from app.graph.state import ResearchState
from app.ingest_store import JobStatus, get_job_store
from app.ingestion.loaders import MAX_PDF_SIZE_BYTES
from app.ingestion.runner import run_ingestion_and_publish
from app.store import RunStatus, get_run_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/research/query", response_model=QueryResponse, status_code=202)
async def start_research(
    payload: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
) -> QueryResponse:
    run_id = str(uuid.uuid4())
    await get_run_store().create_run(run_id, user_id=user_id)

    initial_state: ResearchState = {
        "query": payload.query,
        "user_id": user_id,
        "run_id": run_id,
        "chunks": [],
        "reflection_history": [],
        "cycle_count": 0,
        "tokens_used": 0,
    }

    # BackgroundTasks fires AFTER the response is sent — the client
    # gets run_id immediately and can open the SSE connection right
    # away, matching the "streaming is critical for UX" design.
    background_tasks.add_task(run_graph_and_publish, run_id, initial_state)

    return QueryResponse(run_id=run_id, status=RunStatus.RUNNING.value)


@router.get("/research/{run_id}/stream")
async def stream_research(
    run_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    if not await get_run_store().owns(run_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="run not found")

    return StreamingResponse(
        event_stream_for_run(run_id, request),
        media_type="text/event-stream",
        headers={
            # Prevents intermediary buffering (e.g. some proxy
            # configs) from delaying delivery of SSE chunks.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/research/{run_id}/result", response_model=ResultResponse)
async def get_research_result(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ResultResponse:
    store = get_run_store()

    if not await store.owns(run_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="run not found")

    status = await store.get_status(run_id)
    result = await store.get_result(run_id) or {}

    return ResultResponse(
        run_id=run_id,
        status=status.value if status else RunStatus.RUNNING.value,
        answer=result.get("answer"),
        citations=result.get("citations"),
        cycle_count=result.get("cycle_count"),
        error=result.get("error"),
    )


@router.post("/ingest/document", response_model=IngestResponse, status_code=202)
async def start_ingest(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    text: str | None = Form(None),
) -> IngestResponse:
    """Accepts exactly one of: a PDF file upload, a `url` form field,
    or a `text` form field — matching the three source_type values
    ingest_source() supports. Returns a job_id immediately (per the
    original ingestion design: "user gets a job ID, polls
    /ingest/{job_id}/status") and does the actual parsing/embedding/
    upsert work in the background, since a 20-page PDF can take
    several seconds — far too slow to hold the request open for.
    """
    provided = [v for v in (file, url, text) if v is not None]
    if len(provided) != 1:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of: file (PDF upload), url, or text.",
        )

    if file is not None:
        # Fail fast on grossly oversized uploads before even reading
        # the body into memory — the definitive size check still
        # happens inside load_pdf() too (defense in depth), but this
        # avoids buffering a huge upload just to reject it a moment
        # later. file.size may be None for some transfer encodings,
        # in which case we fall through to the check inside load_pdf.
        if file.size is not None and file.size > MAX_PDF_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_PDF_SIZE_BYTES // (1024 * 1024)}MB limit.",
            )
        source_type = "pdf"
        source: bytes | str = await file.read()
        filename = file.filename
    elif url is not None:
        source_type = "url"
        source = url
        filename = None
    else:
        source_type = "text"
        source = text  # type: ignore[assignment]
        filename = None

    job_id = str(uuid.uuid4())
    await get_job_store().create_job(job_id, user_id=user_id)

    background_tasks.add_task(
        run_ingestion_and_publish,
        job_id,
        source_type=source_type,
        source=source,
        user_id=user_id,
        filename=filename,
    )

    return IngestResponse(job_id=job_id, status=JobStatus.PROCESSING.value)


@router.get("/ingest/{job_id}/status", response_model=IngestStatusResponse)
async def get_ingest_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
) -> IngestStatusResponse:
    store = get_job_store()

    if not await store.owns(job_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="job not found")

    status = await store.get_status(job_id)
    result = await store.get_result(job_id) or {}

    return IngestStatusResponse(
        job_id=job_id,
        status=status.value if status else JobStatus.PROCESSING.value,
        title=result.get("title"),
        source=result.get("source"),
        chunks_ingested=result.get("chunks_ingested"),
        chunks_dropped=result.get("chunks_dropped"),
        error=await store.get_error(job_id),
    )
