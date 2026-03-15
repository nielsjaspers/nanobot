"""Path helpers for Mike storage."""

from __future__ import annotations

from pathlib import Path

from mike.common import ensure_dir, safe_filename


def session_root(data_dir: Path, session_key: str) -> Path:
    return ensure_dir(data_dir / "sessions" / safe_filename(session_key.replace(":", "_")))


def task_root(data_dir: Path, task_id: str) -> Path:
    return ensure_dir(data_dir / "tasks" / safe_filename(task_id))


def history_root(data_dir: Path) -> Path:
    return ensure_dir(data_dir / "history")


def history_records_root(data_dir: Path) -> Path:
    return ensure_dir(history_root(data_dir) / "records")
