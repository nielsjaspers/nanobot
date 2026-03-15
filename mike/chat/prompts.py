"""Prompt building for Mike."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def build_system_prompt(chat_root: Path, skills_summary: str = "") -> str:
    soul = _read(chat_root / "SOUL.md")
    user = _read(chat_root / "USER.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "# Mike",
        "You are Mike, a focused personal assistant bot.",
        f"Current local time: {now}",
        "Use native tool calls for normal work. Use OpenCode delegation only for large code tasks or research.",
        "Shared memory and archived chat history are not loaded into context automatically.",
        "Use the memory/history tools only when you actually need durable recall from past conversations.",
        "If a skill is relevant, read its `SKILL.md` file before using it.",
    ]
    if soul:
        parts.extend(["", "## SOUL", soul])
    if user:
        parts.extend(["", "## USER", user])
    if skills_summary:
        parts.extend(["", "## Skills", skills_summary])
    return "\n".join(parts)
