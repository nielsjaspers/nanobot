"""Web tools for Mike."""

from __future__ import annotations

import html
import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from mike.opencode.search import OpencodeSearch
from mike.tools.base import Tool

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{parsed.scheme or 'none'}'"
        if not parsed.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as exc:
        return False, str(exc)


class WebSearchTool(Tool):
    def __init__(
        self,
        max_results: int = 5,
        cli_bin: str = "opencode",
        attach_url: str = "http://127.0.0.1:4096",
        provider_id: str = "opencode-go",
    ):
        self.max_results = max_results
        self._exa = OpencodeSearch(cli_bin=cli_bin, provider_id=provider_id)
        self._exa.attach_url = attach_url

    def set_context(self, channel: str, chat_id: str, model: str | None = None) -> None:
        del channel, chat_id
        self._exa.set_context(model=model)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web via OpenCode Exa. Returns titles, URLs, and snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        return await self._exa.execute(
            query=query, count=min(max(count or self.max_results, 1), 10)
        )


class WebFetchTool(Tool):
    def __init__(self, max_chars: int = 50000, proxy: str | None = None):
        self.max_chars = max_chars
        self.proxy = proxy

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a URL and extract readable content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extractMode": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "default": "markdown",
                },
                "maxChars": {"type": "integer", "minimum": 100},
            },
            "required": ["url"],
        }

    async def execute(
        self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any
    ) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars
        valid, error = _validate_url(url)
        if not valid:
            return json.dumps(
                {"error": f"URL validation failed: {error}", "url": url}, ensure_ascii=False
            )
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
                proxy=self.proxy,
            ) as client:
                response = await client.get(url, headers={"User-Agent": USER_AGENT})
                response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            if "application/json" in ctype:
                text, extractor = json.dumps(response.json(), indent=2, ensure_ascii=False), "json"
            elif "text/html" in ctype or response.text[:256].lower().startswith(
                ("<!doctype", "<html")
            ):
                doc = Document(response.text)
                content = (
                    self._to_markdown(doc.summary())
                    if extractMode == "markdown"
                    else _strip_tags(doc.summary())
                )
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = response.text, "raw"
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(response.url),
                    "status": response.status_code,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.error("WebFetch error for {}: {}", url, exc)
            return json.dumps({"error": str(exc), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html_text: str) -> str:
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
            lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
            html_text,
            flags=re.I,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
            lambda m: f"\n{'#' * int(m[1])} {_strip_tags(m[2])}\n",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"<li[^>]*>([\s\S]*?)</li>", lambda m: f"\n- {_strip_tags(m[1])}", text, flags=re.I
        )
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return _normalize(_strip_tags(text))
