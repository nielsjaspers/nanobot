"""Bootstrap data files for Mike."""

from __future__ import annotations

from pathlib import Path

from mike.common import ensure_dir
from mike.config import MikeConfig


DEFAULT_SOUL = "# SOUL\n\nYou are Mike, a focused personal assistant bot.\n"
DEFAULT_USER = "# USER\n\nDescribe the owner, preferences, and standing instructions here.\n"
DEFAULT_MEMORY = (
    "# Long-term Memory\n\n"
    "This file stores durable information the bot may want to reference later.\n\n"
    "## User Information\n\n(Important facts about the user)\n\n"
    "## Preferences\n\n(User preferences learned over time)\n\n"
    "## Project Context\n\n(Information about ongoing projects)\n\n"
    "## Important Notes\n\n(Things to remember)\n"
)


def ensure_root(config: MikeConfig) -> Path:
    root = ensure_dir(config.data_dir_path)
    ensure_dir(root / "sessions")
    ensure_dir(root / "history")
    ensure_dir(root / "history" / "records")
    ensure_dir(root / "tasks")
    ensure_dir(root / "logs")
    ensure_dir(root / "skills")
    ensure_shared_files(root)
    seed_research_skill(config, root)
    return root


def ensure_shared_files(root: Path) -> None:
    files = {
        "SOUL.md": DEFAULT_SOUL,
        "USER.md": DEFAULT_USER,
        "MEMORY.md": DEFAULT_MEMORY,
    }
    for name, content in files.items():
        path = root / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    index = root / "history" / "index.json"
    if not index.exists():
        index.write_text("[]\n", encoding="utf-8")


def ensure_session_dirs(root: Path) -> None:
    ensure_dir(root / "uploads")


def seed_research_skill(config: MikeConfig, root: Path) -> None:
    target = root / "skills" / "deep-research" / "SKILL.md"
    if target.exists():
        return
    source = Path(config.deep_research_skill_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return
    target.write_text("# Deep Research\n\nFollow a rigorous research process.\n", encoding="utf-8")
