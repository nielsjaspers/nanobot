"""Search helpers for Mike history archives and memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArchiveMatch:
    archive_id: str
    title: str
    summary: str
    archived_at: str
    metadata: dict[str, Any]
    score: int


def load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def score_entry(query: str, entry: dict[str, Any]) -> int:
    hay_title = str(entry.get("title", ""))
    hay_summary = str(entry.get("summary", ""))
    haystack = f"{hay_title}\n{hay_summary}".lower()
    score = 0
    for token in tokenize(query):
        if token in hay_title.lower():
            score += 5
        if token in hay_summary.lower():
            score += 2
    if query.lower().strip() in haystack:
        score += 8
    return score


def search_index(path: Path, query: str, limit: int = 5) -> list[ArchiveMatch]:
    if not query.strip():
        return []
    results: list[ArchiveMatch] = []
    for entry in load_index(path):
        score = score_entry(query, entry)
        if score <= 0:
            continue
        results.append(
            ArchiveMatch(
                archive_id=str(entry.get("id", "")),
                title=str(entry.get("title", "")),
                summary=str(entry.get("summary", "")),
                archived_at=str(entry.get("archived_at", "")),
                metadata=dict(entry.get("metadata", {}) or {}),
                score=score,
            )
        )
    return sorted(results, key=lambda item: (item.score, item.archived_at), reverse=True)[:limit]


def search_memory_sections(path: Path, query: str, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if not query.strip():
        return text[:max_chars]
    sections = re.split(r"\n(?=## )", text)
    scored: list[tuple[int, str]] = []
    tokens = tokenize(query)
    for section in sections:
        lower = section.lower()
        score = sum(1 for token in tokens if token in lower)
        if score:
            scored.append((score, section.strip()))
    if not scored:
        return ""
    scored.sort(key=lambda item: item[0], reverse=True)
    result = "\n\n".join(section for _, section in scored)
    return result[:max_chars]
