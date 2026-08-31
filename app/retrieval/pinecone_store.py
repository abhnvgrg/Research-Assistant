from __future__ import annotations

import os

from pinecone import AsyncPinecone

from app.retrieval.base import VectorMatch, VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self, index_host: str | None = None, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._index_host = index_host or os.environ.get("PINECONE_INDEX_HOST")
        self._client: AsyncPinecone | None = None
        self._index = None

    async def _get_index(self):
        if self._index is None:
            self._client = AsyncPinecone(api_key=self._api_key)
            self._index = self._client.IndexAsyncio(host=self._index_host)
        return self._index

    async def query(
        self, *, vector: list[float], top_k: int, namespace: str
    ) -> list[VectorMatch]:
        index = await self._get_index()

        response = await index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

        matches: list[VectorMatch] = []
        for match in response.matches:
            metadata = match.metadata or {}
            matches.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "score": match.score,
                }
            )
        return matches

    async def upsert(
        self,
        *,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str,
    ) -> None:
        index = await self._get_index()
        payload = [
            {"id": vec_id, "values": embedding, "metadata": metadata}
            for vec_id, embedding, metadata in vectors
        ]
        await index.upsert(vectors=payload, namespace=namespace)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
