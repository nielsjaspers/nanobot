"""Task persistence for Mike."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mike.common import ensure_dir, timestamp
from mike.storage.files import task_root


@dataclass
class TaskRecord:
    task_id: str
    session_key: str
    origin_channel: str
    origin_chat_id: str
    kind: str
    query: str
    title: str
    backend: str = "opencode"
    status: str = "queued"
    phase: str = "RECEIVED"
    model: str | None = None
    progress_summary: str | None = None
    final_summary: str | None = None
    final_artifact: str | None = None
    backend_session_id: str | None = None
    user_injections: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)

    def to_dict(self) -> dict[str, Any]:
        self.updated_at = timestamp()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        return cls(**data)


class TaskStore:
    def __init__(self, data_dir: Path):
        self.data_dir = ensure_dir(data_dir)

    def root(self, task_id: str) -> Path:
        return ensure_dir(self.data_dir / task_id)

    def snapshot_path(self, task_id: str) -> Path:
        return self.root(task_id) / "task.json"

    def events_path(self, task_id: str) -> Path:
        return self.root(task_id) / "events.jsonl"

    def injections_path(self, task_id: str) -> Path:
        return self.root(task_id) / "injections.jsonl"

    def artifacts_dir(self, task_id: str) -> Path:
        return ensure_dir(self.root(task_id) / "artifacts")

    def save(self, task: TaskRecord) -> None:
        self.snapshot_path(task.task_id).write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        entry = {"timestamp": timestamp(), "type": event_type, "payload": payload}
        with self.events_path(task_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_injection(self, task_id: str, text: str) -> None:
        with self.injections_path(task_id).open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"timestamp": timestamp(), "text": text}, ensure_ascii=False) + "\n"
            )

    def load(self, task_id: str) -> TaskRecord | None:
        path = self.snapshot_path(task_id)
        if not path.exists():
            return None
        return TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[TaskRecord]:
        items: list[TaskRecord] = []
        for path in self.data_dir.glob("*/task.json"):
            try:
                items.append(TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        legacy_root = self.data_dir / "tasks"
        if legacy_root.exists():
            for path in legacy_root.glob("*/task.json"):
                try:
                    items.append(TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
                except Exception:
                    continue
        deduped: dict[str, TaskRecord] = {item.task_id: item for item in items}
        return sorted(deduped.values(), key=lambda item: item.updated_at, reverse=True)

    def write_artifact(self, task_id: str, name: str, content: str) -> str:
        path = self.artifacts_dir(task_id) / name
        path.write_text(content, encoding="utf-8")
        return str(path)
