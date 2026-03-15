"""Compatibility helpers for structured history archives."""

from __future__ import annotations

from pathlib import Path

from mike.memory.search import search_index


def search_history(path: Path, query: str, limit: int = 20) -> list[str]:
    matches = search_index(path, query, limit=limit)
    return [f"{match.archive_id}: {match.title} - {match.summary}" for match in matches]
