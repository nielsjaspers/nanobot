"""Chat state persistence for Mike."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mike.bootstrap import ensure_root, ensure_session_dirs
from mike.common import ensure_dir, safe_filename, timestamp
from mike.config import MikeConfig
from mike.storage.files import history_records_root, history_root, session_root


@dataclass
class ChatSession:
    key: str
    current_model: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)

    def add_message(self, role: str, content: Any, **extra: Any) -> None:
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **extra}
        )
        self.updated_at = timestamp()

    def history(self, limit: int = 500) -> list[dict[str, Any]]:
        items = self.messages[-limit:]
        result: list[dict[str, Any]] = []
        for item in items:
            clean = {"role": item["role"], "content": item.get("content")}
            for key in (
                "tool_calls",
                "tool_call_id",
                "name",
                "reasoning_content",
                "thinking_blocks",
            ):
                if key in item:
                    clean[key] = item[key]
            result.append(clean)
        return result

    def clear(self) -> None:
        self.messages = []
        now = timestamp()
        self.created_at = now
        self.updated_at = now

    def has_meaningful_content(self) -> bool:
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "system"} and content:
                return True
        return False


class ChatStore:
    def __init__(self, config: MikeConfig):
        self.config = config
        self.data_dir = ensure_root(config)
        self._cache: dict[str, ChatSession] = {}

    @property
    def shared_root(self) -> Path:
        return self.data_dir

    def session_root(self, session_key: str) -> Path:
        root = session_root(self.data_dir, session_key)
        ensure_session_dirs(root)
        return root

    def state_path(self, session_key: str) -> Path:
        return self.session_root(session_key) / "active.json"

    def uploads_dir(self, session_key: str) -> Path:
        return ensure_dir(self.session_root(session_key) / "uploads")

    def shared_file(self, name: str) -> Path:
        return self.shared_root / name

    def memory_path(self) -> Path:
        return self.shared_file("MEMORY.md")

    def soul_path(self) -> Path:
        return self.shared_file("SOUL.md")

    def user_path(self) -> Path:
        return self.shared_file("USER.md")

    def skills_root(self) -> Path:
        return self.shared_root / "skills"

    def history_index_path(self) -> Path:
        return history_root(self.data_dir) / "index.json"

    def history_records_root(self) -> Path:
        return history_records_root(self.data_dir)

    def history_record_path(self, archive_id: str) -> Path:
        return self.history_records_root() / f"{safe_filename(archive_id)}.json"

    def get(self, session_key: str) -> ChatSession:
        if session_key in self._cache:
            return self._cache[session_key]
        path = self.state_path(session_key)
        if not path.exists():
            session = ChatSession(key=session_key)
            self.save(session)
            return session
        data = json.loads(path.read_text(encoding="utf-8"))
        session = ChatSession(**data)
        self._cache[session_key] = session
        return session

    def save(self, session: ChatSession) -> None:
        path = self.state_path(session.key)
        payload = {
            "key": session.key,
            "current_model": session.current_model,
            "messages": session.messages,
            "created_at": session.created_at,
            "updated_at": timestamp(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache[session.key] = session

    def reset(self, session_key: str, preserve_model: bool = True) -> ChatSession:
        current = (
            self.get(session_key)
            if session_key in self._cache or self.state_path(session_key).exists()
            else None
        )
        session = ChatSession(
            key=session_key,
            current_model=current.current_model if preserve_model and current else None,
        )
        self.save(session)
        self._cache[session_key] = session
        return session

    def save_upload(self, session_key: str, filename: str, data: bytes) -> str:
        safe = safe_filename(filename or "upload.bin")
        path = self.uploads_dir(session_key) / safe
        path.write_bytes(data)
        return str(path)
