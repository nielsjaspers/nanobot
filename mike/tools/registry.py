"""Native tool registry for Mike."""

from __future__ import annotations

from typing import Any

from mike.tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        hint = "\n\n[Analyze the error above and try a different approach.]"
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self._tools)}"
        try:
            cast = tool.cast_params(params)
            errors = tool.validate_params(cast)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + hint
            result = await tool.execute(**cast)
            if isinstance(result, str) and result.startswith("Error"):
                return result + hint
            return result
        except Exception as exc:
            return f"Error executing {name}: {exc}" + hint
