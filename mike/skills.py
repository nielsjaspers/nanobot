"""Skill discovery for Mike."""

from __future__ import annotations

from pathlib import Path


def list_skills(chat_root: Path) -> list[Path]:
    skills_dir = chat_root / "skills"
    if not skills_dir.exists():
        return []
    return sorted(path for path in skills_dir.glob("*/SKILL.md") if path.is_file())


def build_summary(chat_root: Path) -> str:
    lines: list[str] = []
    for path in list_skills(chat_root):
        name = path.parent.name
        first = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("---") and not text.startswith("#"):
                first = text
                break
        lines.append(f"- {name}: {first or 'Read the SKILL.md file for details.'} ({path})")
    return "\n".join(lines)
