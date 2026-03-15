"""Shell execution tool for Mike."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from mike.tools.base import Tool


class ExecTool(Tool):
    def __init__(
        self,
        timeout: int = 120,
        working_dir: str | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.restrict_to_workspace = restrict_to_workspace
        self.deny_patterns = [
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fq]\b",
            r"\brmdir\s+/s\b",
            r"(?:^|[;&|]\s*)format\b",
            r"\b(mkfs|diskpart)\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\b(shutdown|reboot|poweroff)\b",
            r":\(\)\s*\{.*\};\s*:",
        ]

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        if not command.strip():
            return "Error: Missing required parameter: command"
        cwd = working_dir or self.working_dir or os.getcwd()
        guard = self._guard(command, cwd)
        if guard:
            return guard
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout or self.timeout)
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout or self.timeout} seconds"
        except Exception as exc:
            return f"Error executing command: {exc}"
        parts = []
        if out:
            parts.append(out.decode("utf-8", errors="replace"))
        if err:
            text = err.decode("utf-8", errors="replace")
            if text.strip():
                parts.append("STDERR:\n" + text)
        parts.append(f"\nExit code: {proc.returncode}")
        return "\n".join(parts) if parts else "(no output)"

    def _guard(self, command: str, cwd: str) -> str | None:
        lowered = command.lower()
        for pattern in self.deny_patterns:
            if re.search(pattern, lowered):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"
        if self.restrict_to_workspace and ("../" in command or "..\\" in command):
            return "Error: Command blocked by safety guard (path traversal detected)"
        if self.restrict_to_workspace:
            base = Path(cwd).resolve()
            for raw in re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command):
                try:
                    path = Path(raw).expanduser().resolve()
                except Exception:
                    continue
                if path.is_absolute() and base not in path.parents and path != base:
                    return "Error: Command blocked by safety guard (path outside working dir)"
        return None
