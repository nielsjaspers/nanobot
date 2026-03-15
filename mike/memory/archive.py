"""Archive pipeline for Mike conversations."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import json_repair
from loguru import logger

from mike.chat.models import clamp_max_tokens
from mike.common import timestamp
from mike.memory.search import load_index
from mike.storage.chats import ChatSession, ChatStore
from mike.llm import LLMProvider

ARCHIVE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "archive_conversation",
            "description": "Save conversation archival results for Mike.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "memory_update": {"type": "string"},
                },
                "required": ["title", "summary", "memory_update"],
            },
        },
    }
]


@dataclass
class ArchiveResult:
    archive_id: str
    title: str
    summary: str


class ArchiveManager:
    def __init__(self, store: ChatStore, provider: LLMProvider, model_getter):
        self.store = store
        self.provider = provider
        self.model_getter = model_getter

    async def archive_session(
        self, session: ChatSession, *, channel: str, chat_id: str
    ) -> ArchiveResult:
        if not session.has_meaningful_content():
            raise RuntimeError("No conversation content to archive.")
        current_memory = self.store.memory_path().read_text(encoding="utf-8")
        transcript = self._format_messages(session.messages)
        prompt = (
            "Archive this conversation for Mike. Respond ONLY with valid JSON. "
            "The JSON must contain exactly these string keys: title, summary, memory_update. "
            "The summary should be useful for later search and recall.\n\n"
            f"## Current MEMORY.md\n{current_memory}\n\n"
            f"## Conversation\n{transcript}"
        )
        title, summary, memory_update = await self._summarize_archive(
            session=session,
            prompt=prompt,
            current_memory=current_memory,
        )
        self.store.memory_path().write_text(memory_update, encoding="utf-8")
        archive_id = str(uuid.uuid4())[:12]
        metadata = self._build_metadata(session, channel=channel, chat_id=chat_id)
        record = {
            "id": archive_id,
            "title": title,
            "summary": summary,
            "full_chat_log": session.messages,
            "metadata": metadata,
        }
        self.store.history_record_path(archive_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index = load_index(self.store.history_index_path())
        index.append(
            {
                "id": archive_id,
                "title": title,
                "summary": summary,
                "archived_at": metadata["archived_at"],
                "metadata": {
                    "channel": metadata["channel"],
                    "chat_id": metadata["chat_id"],
                    "message_count": metadata["message_count"],
                    "models_used": metadata["models_used"],
                    "started_at": metadata["started_at"],
                    "ended_at": metadata["ended_at"],
                },
            }
        )
        self.store.history_index_path().write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ArchiveResult(archive_id=archive_id, title=title, summary=summary)

    async def _summarize_archive(
        self,
        *,
        session: ChatSession,
        prompt: str,
        current_memory: str,
    ) -> tuple[str, str, str]:
        try:
            response = await self.provider.chat_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Mike's archival assistant. Respond only with JSON containing "
                            "title, summary, and memory_update."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model_getter(session),
                max_tokens=clamp_max_tokens(self.model_getter(session), 4096),
                temperature=0.1,
            )
            parsed = self._parse_archive_json(response.content)
            if parsed:
                title = str(parsed.get("title", "")).strip()
                summary = str(parsed.get("summary", "")).strip()
                memory_update = str(parsed.get("memory_update", "")).strip()
                if title and summary and memory_update:
                    return title, summary, memory_update
            logger.warning(
                "Archive summarizer returned invalid JSON; using fallback archive summary"
            )
        except Exception as exc:
            logger.warning("Archive summarizer failed: {}", exc)
        return self._fallback_summary(session, current_memory)

    @staticmethod
    def _parse_archive_json(content: str | None) -> dict[str, Any] | None:
        if not content:
            return None
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json\n", "", 1).strip()
        try:
            data = json_repair.loads(raw)
        except Exception:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    data = json_repair.loads(raw[start : end + 1])
                else:
                    return None
            except Exception:
                return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _fallback_summary(session: ChatSession, current_memory: str) -> tuple[str, str, str]:
        user_messages = [message for message in session.messages if message.get("role") == "user"]
        assistant_messages = [
            message for message in session.messages if message.get("role") == "assistant"
        ]
        first_user = ""
        if user_messages:
            content = user_messages[0].get("content")
            first_user = (
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            )
        last_assistant = ""
        if assistant_messages:
            content = assistant_messages[-1].get("content")
            last_assistant = (
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            )
        title = (
            first_user.strip().splitlines()[0][:80]
            if first_user.strip()
            else "Conversation archive"
        )
        summary = f"Archived conversation with {len(user_messages)} user messages and {len(assistant_messages)} assistant messages."
        if first_user.strip():
            summary += f" Started with: {first_user.strip()[:160]}"
        if last_assistant.strip():
            summary += f" Last reply: {last_assistant.strip()[:160]}"
        return title or "Conversation archive", summary, current_memory

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = str(message.get("role", "?")).upper()
            content = message.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                rendered = json.dumps(content, ensure_ascii=False)
            elif isinstance(content, dict):
                rendered = json.dumps(content, ensure_ascii=False)
            else:
                rendered = str(content)
            lines.append(f"[{message.get('timestamp', '?')[:16]}] {role}: {rendered}")
        return "\n".join(lines)

    @staticmethod
    def _build_metadata(session: ChatSession, *, channel: str, chat_id: str) -> dict[str, Any]:
        models = []
        if session.current_model:
            models.append(session.current_model)
        return {
            "channel": channel,
            "chat_id": chat_id,
            "session_key": session.key,
            "message_count": len(session.messages),
            "models_used": sorted(set(models)),
            "started_at": session.created_at,
            "ended_at": session.updated_at,
            "archived_at": timestamp(),
        }
