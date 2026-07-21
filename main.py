"""
FastAPI application entrypoint.

The startup event fires the Pinecone warmup ping (fix option from
the tool-scrutiny session — "warmup ping on app start", the
highest-value/lowest-effort fix against serverless cold start
latency), and initializes the LangGraph checkpointer so research runs
become resumable across process restarts. Both are wrapped so a
failure never crashes app startup — they're optimizations/resilience
features, not hard dependencies, matching the "never let
observability/optimization code break the critical path" principle
applied consistently across the design.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db import close_pool, init_pool
from app.graph.checkpointing import close_checkpointer, init_checkpointer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.retrieval.pinecone_store import PineconeVectorStore

        store = PineconeVectorStore()
        await store.query(vector=[0.0] * 1536, top_k=1, namespace="__warmup__")
        logger.info("Pinecone warmup succeeded")
    except Exception as e:  # pragma: no cover - best-effort only
        logger.warning("Pinecone warmup failed (non-fatal): %s", e)

    await init_checkpointer()
    await init_pool()  # RunStore/JobStore/QuotaStore fall back to in-memory if this doesn't succeed

    yield  # app runs here

    await close_checkpointer()
    await close_pool()


app = FastAPI(title="Research Assistant API", lifespan=lifespan)

# Permissive for local development — the frontend (single static HTML
# file, opened via file:// or a local static server) needs to call
# this API from a different origin. Tighten to your real frontend's
# origin before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
