"""Telegram rendering helpers for Mike."""

from __future__ import annotations

import re
import unicodedata


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _render_table_box(table_lines: list[str]) -> str:
    def dw(value: str) -> int:
        return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in value)

    rows: list[list[str]] = []
    has_sep = False
    for line in table_lines:
        cells = [_strip_md(cell) for cell in line.strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", cell) for cell in cells if cell):
            has_sep = True
            continue
        rows.append(cells)
    if not rows or not has_sep:
        return "\n".join(table_lines)
    cols = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (cols - len(row)))
    widths = [max(dw(row[idx]) for row in rows) for idx in range(cols)]

    def draw(cells: list[str]) -> str:
        return "  ".join(f"{cell}{' ' * (width - dw(cell))}" for cell, width in zip(cells, widths))

    output = [draw(rows[0]), "  ".join("-" * width for width in widths)]
    for row in rows[1:]:
        output.append(draw(row))
    return "\n".join(output)


def markdown_to_telegram_html(text: str) -> str:
    if not text:
        return ""
    code_blocks: list[str] = []

    def save_code_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)
    lines = text.split("\n")
    rebuilt: list[str] = []
    idx = 0
    while idx < len(lines):
        if re.match(r"^\s*\|.+\|", lines[idx]):
            table: list[str] = []
            while idx < len(lines) and re.match(r"^\s*\|.+\|", lines[idx]):
                table.append(lines[idx])
                idx += 1
            box = _render_table_box(table)
            if box != "\n".join(table):
                code_blocks.append(box)
                rebuilt.append(f"\x00CB{len(code_blocks) - 1}\x00")
            else:
                rebuilt.extend(table)
            continue
        rebuilt.append(lines[idx])
        idx += 1
    text = "\n".join(rebuilt)

    inline_codes: list[str] = []

    def save_inline_code(match: re.Match[str]) -> str:
        inline_codes.append(match.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)
    for idx, code in enumerate(inline_codes):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{idx}\x00", f"<code>{escaped}</code>")
    for idx, code in enumerate(code_blocks):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{idx}\x00", f"<pre><code>{escaped}</code></pre>")
    return text
