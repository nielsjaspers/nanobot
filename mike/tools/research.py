"""Native research tool for Mike."""

from __future__ import annotations

from typing import Any

from mike.tools.base import Tool


class ResearchTool(Tool):
    def __init__(self, manager: Any):
        self._manager = manager
        self._channel = "cli"
        self._chat_id = "direct"
        self._session_key = "cli:direct"
        self._model: str | None = None

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._model = model

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return "Start or manage a background research task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "status", "cancel"]},
                "query": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(
        self, action: str, query: str | None = None, task_id: str | None = None, **kwargs: Any
    ) -> str:
        if action == "start":
            if not query or not query.strip():
                return "Error: query is required when action=start"
            return await self._manager.start_task(
                query=query,
                session_key=self._session_key,
                channel=self._channel,
                chat_id=self._chat_id,
                model=self._model,
                kind="research",
            )
        if action == "status":
            return self._manager.format_status(self._session_key)
        if action == "cancel":
            if task_id:
                await self._manager.cancel_task(task_id)
                return f"Cancelled research task {task_id}."
            count = await self._manager.cancel_by_session(self._session_key)
            return (
                f"Cancelled {count} research task(s)."
                if count
                else "No running research task found for this chat."
            )
        return f"Error: Unsupported action '{action}'"
