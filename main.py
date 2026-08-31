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
    await init_pool()

    yield

    await close_checkpointer()
    await close_pool()


app = FastAPI(title="Research Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
