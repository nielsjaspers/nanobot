"""Shared memory access tools for Mike."""

from __future__ import annotations

from typing import Any

from mike.memory.search import search_memory_sections
from mike.tools.base import Tool


class ReadMemoryTool(Tool):
    def __init__(self, memory_path_getter):
        self.memory_path_getter = memory_path_getter

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return "Read shared MEMORY.md only when durable memory is needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 200},
            },
        }

    async def execute(self, query: str | None = None, max_chars: int = 4000, **kwargs: Any) -> str:
        path = self.memory_path_getter()
        if not path.exists():
            return "No shared MEMORY.md file exists yet."
        if query and query.strip():
            result = search_memory_sections(path, query, max_chars=max_chars)
            return result or f"No memory entries matched query: {query}"
        return path.read_text(encoding="utf-8")[:max_chars]
