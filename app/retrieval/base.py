from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class VectorMatch(TypedDict):
    text: str
    source: str
    title: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def query(
        self, *, vector: list[float], top_k: int, namespace: str
    ) -> list[VectorMatch]:
        pass

    @abstractmethod
    async def upsert(
        self,
        *,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str,
    ) -> None:
        pass
