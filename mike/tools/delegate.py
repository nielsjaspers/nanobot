"""Native delegation tool that hands large tasks to OpenCode."""

from __future__ import annotations

from typing import Any

from mike.tools.base import Tool


class OpenCodeDelegateTool(Tool):
    def __init__(self, manager: Any):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._model: str | None = None

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._model = model

    @property
    def name(self) -> str:
        return "opencode_delegate"

    @property
    def description(self) -> str:
        return "Delegate a complex task to OpenCode. Use this for large or autonomous subtasks."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        if not task.strip():
            return "Error: Missing required parameter: task"
        return await self._manager.start_task(
            query=task,
            session_key=self._session_key,
            channel=self._origin_channel,
            chat_id=self._origin_chat_id,
            title=label or task[:40].strip() or "task",
            model=self._model,
            kind="delegate",
        )
