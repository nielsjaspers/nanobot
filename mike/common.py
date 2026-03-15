"""Shared helpers for Mike."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from mike.helpers import detect_image_mime, split_message

UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().isoformat()


def safe_filename(name: str) -> str:
    return UNSAFE_CHARS.sub("_", name).strip()


def json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


__all__ = [
    "detect_image_mime",
    "ensure_dir",
    "json_dump",
    "safe_filename",
    "split_message",
    "timestamp",
]
