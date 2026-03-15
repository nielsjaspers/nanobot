"""Structured history archive tools for Mike."""

from __future__ import annotations

import json
from typing import Any

from mike.memory.search import search_index
from mike.tools.base import Tool


class SearchHistoryTool(Tool):
    def __init__(self, index_path_getter):
        self.index_path_getter = index_path_getter

    @property
    def name(self) -> str:
        return "search_history"

    @property
    def description(self) -> str:
        return "Search archived conversation titles and summaries to find relevant past chats."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        matches = search_index(self.index_path_getter(), query, limit=limit)
        if not matches:
            return f"No archived conversations matched query: {query}"
        payload = [
            {
                "archive_id": match.archive_id,
                "title": match.title,
                "summary": match.summary,
                "archived_at": match.archived_at,
                "metadata": match.metadata,
            }
            for match in matches
        ]
        return json.dumps(payload, ensure_ascii=False, indent=2)


class GetHistoryConversationTool(Tool):
    def __init__(self, record_path_getter):
        self.record_path_getter = record_path_getter

    @property
    def name(self) -> str:
        return "get_history_conversation"

    @property
    def description(self) -> str:
        return "Fetch the full archived chat log for one archived conversation id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "archive_id": {"type": "string"},
            },
            "required": ["archive_id"],
        }

    async def execute(self, archive_id: str, **kwargs: Any) -> str:
        path = self.record_path_getter(archive_id)
        if not path.exists():
            return f"No archived conversation found for id: {archive_id}"
        return path.read_text(encoding="utf-8")
