"""
VectorStore interface — the abstraction we specifically scrutinized
as the highest-value, lowest-effort fix against Pinecone vendor
lock-in (see tool_scrutiny design phase). Nodes never import
Pinecone directly; they call through this interface, so swapping to
Qdrant later touches one file (a new implementation of this ABC),
not every call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class VectorMatch(TypedDict):
    """A single retrieval result — deliberately shaped to convert
    trivially into the graph's Chunk type."""

    text: str
    source: str
    title: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def query(
        self, *, vector: list[float], top_k: int, namespace: str
    ) -> list[VectorMatch]:
        """Returns up to top_k matches for the given embedding,
        scoped to namespace. Namespace MUST be derived from a
        verified JWT sub upstream — never from client input."""

    @abstractmethod
    async def upsert(
        self,
        *,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str,
    ) -> None:
        """vectors is a list of (id, embedding, metadata) tuples.
        metadata must include at least 'text' and 'source' — the
        query() implementation reads these back out to build
        VectorMatch objects."""
