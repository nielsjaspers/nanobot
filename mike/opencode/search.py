"""OpenCode Exa wrapper for Mike web search."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any
from urllib.parse import urlparse


class OpencodeSearch:
    def __init__(
        self, cli_bin: str = "opencode", timeout: int = 120, provider_id: str = "opencode-go"
    ):
        self.cli_bin = cli_bin
        self.timeout = timeout
        self.provider_id = provider_id
        self.model: str | None = None
        self.attach_url = "http://127.0.0.1:4096"
        self.agent: str | None = None

    def set_context(
        self, model: str | None = None, attach_url: str | None = None, agent: str | None = None
    ) -> None:
        self.model = model
        if attach_url:
            self.attach_url = attach_url
        self.agent = agent

    async def execute(self, query: str, count: int = 5) -> str:
        if not query.strip():
            return "Error: Missing required parameter: query"
        if not self._is_local_url(self.attach_url):
            return "Error: attach_url must be localhost (privacy policy)"
        limit = min(max(count, 1), 10)
        prompt = (
            "Use the Exa web search tool to search for: "
            f"{query!r}. Return ONLY JSON with this schema: "
            '{"results":[{"title":...,"url":...,"snippet":...}]}. '
            f"Limit to {limit} results. No extra text."
        )
        cmd = [self.cli_bin, "run", "--format", "json", "--attach", self.attach_url]
        if self.model:
            cmd.extend(["--model", f"{self.provider_id}/{self.model}"])
        if self.agent:
            cmd.extend(["--agent", self.agent])
        cmd.append(prompt)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return f"Error: Timed out after {self.timeout} seconds"
        except FileNotFoundError:
            return f"Error: '{self.cli_bin}' not found. Install OpenCode or set the correct binary."
        stdout = out.decode("utf-8", errors="replace") if out else ""
        stderr = err.decode("utf-8", errors="replace") if err else ""
        if proc.returncode not in (0, None):
            return f"Error: {(stdout or stderr).strip() or 'opencode run failed'}"
        data = self._extract_json(stdout)
        return (
            json.dumps(data, ensure_ascii=False)
            if data is not None
            else stdout.strip() or "Error: empty response from opencode"
        )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        results = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "tool_use":
                part = event.get("part", {})
                if (
                    part.get("tool") == "websearch"
                    and part.get("state", {}).get("status") == "completed"
                ):
                    output = part.get("state", {}).get("output", "")
                    if output:
                        results.extend(self._parse_websearch_output(output))
            elif "results" in event:
                return event
        if results:
            return {"results": results}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _parse_websearch_output(output: str) -> list[dict[str, str]]:
        pattern = r"Title:\s*(.+?)\n(?:Published Date:\s*(.+?)\n)?(?:Author:\s*(.+?)\n)?URL:\s*(.+?)\nText:\s*(.+?)(?=\n\nTitle:|$)"
        items = []
        for title, _published, _author, url, text in re.findall(pattern, output, re.DOTALL):
            snippet = text.strip()
            items.append(
                {
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": snippet[:300] + ("..." if len(snippet) > 300 else ""),
                }
            )
        return items

    @staticmethod
    def _is_local_url(url: str) -> bool:
        try:
            host = (urlparse(url).hostname or "").lower()
            return host in {"localhost", "127.0.0.1", "::1"}
        except Exception:
            return False
