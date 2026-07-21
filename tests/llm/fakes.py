from __future__ import annotations

from types import SimpleNamespace


def fake_completion(content: str, total_tokens: int | None = None):
    usage = (
        None
        if total_tokens is None
        else SimpleNamespace(total_tokens=total_tokens)
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )

