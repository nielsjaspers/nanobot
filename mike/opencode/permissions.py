"""Permission helpers for OpenCode-backed tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def default_rules() -> list[dict[str, Any]]:
    return [
        {"permission": "question", "action": "deny", "pattern": "*"},
        {"permission": "plan_enter", "action": "deny", "pattern": "*"},
        {"permission": "plan_exit", "action": "deny", "pattern": "*"},
    ]


@dataclass
class PendingPermission:
    request_id: str
    session_id: str
    permission: str
    patterns: list[str]
    task_id: str
    channel: str
    chat_id: str
