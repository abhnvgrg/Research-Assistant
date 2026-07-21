"""
Tavily web search client. Kept as a thin wrapper (not behind an ABC
like VectorStore) because there's only one web search implementation
in this design — the interface pattern earns its cost when a real
second implementation exists, not preemptively everywhere.

search_depth='advanced' per our earlier design: Tavily's basic depth
returns ~200 char snippets; advanced fetches and extracts full
article text, which the synthesizer needs for real citations.
"""

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
    """One Tavily call per sub-question. Returns normalized results —
    'content' -> text, 'url' -> source — so callers never see
    Tavily's raw response shape."""
    client = get_tavily_client()

    response = await client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,  # 'content' field is already the extracted text
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
