from __future__ import annotations

import os
from typing import TypedDict

from tavily import AsyncTavilyClient


class WebResult(TypedDict):
    text: str
    source: str
    title: str
    score: float


_client: AsyncTavilyClient | None = None


def get_tavily_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    return _client


async def search_web(query: str, *, max_results: int = 5) -> list[WebResult]:
    client = get_tavily_client()

    response = await client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,
    )

    results: list[WebResult] = []
    for item in response.get("results", []):
        results.append(
            {
                "text": item.get("content", ""),
                "source": item.get("url", ""),
                "title": item.get("title", ""),
                "score": item.get("score", 0.0),
            }
        )
    return results
