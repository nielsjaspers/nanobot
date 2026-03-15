"""Background task management for Mike."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mike.storage.tasks import TaskRecord, TaskStore


@dataclass
class RunningTask:
    task_id: str
    raw_task: asyncio.Task[None]
    session_key: str


class TaskManager:
    def __init__(self, store: TaskStore):
        self.store = store
        self._running: dict[str, RunningTask] = {}

    def add(self, task_id: str, session_key: str, raw_task: asyncio.Task[None]) -> None:
        self._running[task_id] = RunningTask(
            task_id=task_id, raw_task=raw_task, session_key=session_key
        )
        raw_task.add_done_callback(lambda _: self._running.pop(task_id, None))

    def list(self, session_key: str | None = None) -> list[TaskRecord]:
        tasks = self.store.list()
        if session_key is None:
            return tasks
        return [task for task in tasks if task.session_key == session_key]

    async def cancel_task(self, task_id: str) -> None:
        running = self._running.get(task_id)
        if running and not running.raw_task.done():
            running.raw_task.cancel()
            await asyncio.gather(running.raw_task, return_exceptions=True)

    async def cancel_by_session(self, session_key: str) -> int:
        count = 0
        for task_id, running in list(self._running.items()):
            if running.session_key != session_key:
                continue
            await self.cancel_task(task_id)
            count += 1
        return count
