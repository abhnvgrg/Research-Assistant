from __future__ import annotations

import logging
import os
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
    await init_pool()

    yield

    await close_checkpointer()
    await close_pool()


app = FastAPI(title="Research Assistant API", lifespan=lifespan)
app.include_router(router)

allowed_origins = [origin.strip() for origin in os.environ.get(
    "FRONTEND_ORIGINS",
    "null,http://localhost:3000,http://127.0.0.1:3000",
).split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
