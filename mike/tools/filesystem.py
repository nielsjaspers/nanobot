"""Filesystem tools for Mike."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from mike.tools.base import Tool


def _resolve_path(
    path: str, workspace: Path | None = None, allowed_dir: Path | None = None
) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and workspace:
        value = workspace / value
    resolved = value.resolve()
    if allowed_dir:
        resolved.relative_to(allowed_dir.resolve())
    return resolved


class _FsTool(Tool):
    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self.workspace = workspace
        self.allowed_dir = allowed_dir

    def _resolve(self, path: str) -> Path:
        return _resolve_path(path, self.workspace, self.allowed_dir)


class ReadFileTool(_FsTool):
    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a file with numbered lines. Use offset and limit for large files."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str, offset: int = 1, limit: int | None = None, **kwargs: Any
    ) -> str:
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"
            lines = fp.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            if total == 0:
                return f"(Empty file: {path})"
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"
            start = max(offset - 1, 0)
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + idx + 1}| {line}" for idx, line in enumerate(lines[start:end])]
            result = "\n".join(numbered)
            if len(result) > self._MAX_CHARS:
                result = result[: self._MAX_CHARS] + "\n... (truncated)"
            if end < total:
                return (
                    result
                    + f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
                )
            return result + f"\n\n(End of file - {total} lines total)"
        except Exception as exc:
            return f"Error reading file: {exc}"


class WriteFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file, creating parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            fp = self._resolve(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {fp}"
        except Exception as exc:
            return f"Error writing file: {exc}"


class EditFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_text with new_text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False, **kwargs: Any
    ) -> str:
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            content = fp.read_text(encoding="utf-8")
            if old_text not in content:
                close = difflib.get_close_matches(old_text, content.splitlines(), n=1)
                extra = f" Closest line: {close[0]!r}" if close else ""
                return f"Error: old_text not found in {path}.{extra}"
            updated = (
                content.replace(old_text, new_text)
                if replace_all
                else content.replace(old_text, new_text, 1)
            )
            fp.write_text(updated, encoding="utf-8")
            return f"Successfully edited {fp}"
        except Exception as exc:
            return f"Error editing file: {exc}"


class ListDirTool(_FsTool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories in a path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            root = self._resolve(path)
            if not root.exists():
                return f"Error: Path not found: {path}"
            if root.is_file():
                return f"Error: Not a directory: {path}"
            items = []
            for item in sorted(
                root.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower())
            ):
                items.append(item.name + ("/" if item.is_dir() else ""))
            return "\n".join(items) if items else "(empty directory)"
        except Exception as exc:
            return f"Error listing directory: {exc}"
