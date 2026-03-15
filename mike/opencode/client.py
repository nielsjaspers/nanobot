"""OpenCode Serve client for Mike."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class OpencodeServeError(RuntimeError):
    pass


class OpencodeClient:
    def __init__(
        self,
        base_url: str,
        username: str = "opencode",
        password: str = "",
        directory: str | None = None,
        timeout: int = 1800,
    ):
        auth = (username, password) if password else None
        headers: dict[str, str] = {}
        if directory:
            headers["x-opencode-directory"] = directory
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            auth=auth,
            headers=headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/path")

    async def create_session(
        self,
        title: str | None = None,
        permission: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if permission:
            payload["permission"] = permission
        return await self._request_json("POST", "/session", json=payload)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await self._request_json("GET", f"/session/{session_id}")

    async def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        payload = await self._request_json("GET", f"/session/{session_id}/message")
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return data
        if isinstance(payload, list):
            return payload
        return []

    async def prompt_async(
        self,
        session_id: str,
        text: str,
        *,
        system: str | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        agent: str | None = None,
        no_reply: bool = False,
        reasoning_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if system:
            body["system"] = system
        if provider_id and model_id:
            body["model"] = {"providerID": provider_id, "modelID": model_id}
        if agent:
            body["agent"] = agent
        if no_reply:
            body["noReply"] = True
        if reasoning_config:
            body["reasoningConfig"] = reasoning_config
        return await self._request_json("POST", f"/session/{session_id}/prompt_async", json=body)

    async def prompt(
        self,
        session_id: str,
        text: str,
        *,
        system: str | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        agent: str | None = None,
        no_reply: bool = False,
        reasoning_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if system:
            body["system"] = system
        if provider_id and model_id:
            body["model"] = {"providerID": provider_id, "modelID": model_id}
        if agent:
            body["agent"] = agent
        if no_reply:
            body["noReply"] = True
        if reasoning_config:
            body["reasoningConfig"] = reasoning_config
        return await self._request_json("POST", f"/session/{session_id}/message", json=body)

    async def abort_session(self, session_id: str) -> None:
        await self._request_json("POST", f"/session/{session_id}/abort")

    async def list_permissions(self) -> list[dict[str, Any]]:
        payload = await self._request_json("GET", "/permission")
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return data
        if isinstance(payload, list):
            return payload
        return []

    async def reply_permission(
        self, request_id: str, reply: str, message: str | None = None
    ) -> bool:
        payload: dict[str, Any] = {"reply": reply}
        if message:
            payload["message"] = message
        result = await self._request_json("POST", f"/permission/{request_id}/reply", json=payload)
        if isinstance(result, dict) and "data" in result:
            return bool(result["data"])
        return bool(result)

    async def wait_for_text(
        self, session_id: str, max_polls: int = 900, delay_s: float = 2.0
    ) -> str:
        for _ in range(max_polls):
            items = await self.list_messages(session_id)
            assistants = []
            for message in items:
                if not isinstance(message, dict):
                    continue
                info = message.get("info") or {}
                if info.get("role") == "assistant":
                    assistants.append(message)
            if assistants:
                latest = assistants[-1]
                text = self.extract_text(latest)
                if text:
                    return text
                info = latest.get("info") or {}
                finish = info.get("finish")
                time_info = info.get("time") if isinstance(info, dict) else None
                if finish not in (None, "tool-calls") and isinstance(time_info, dict):
                    if time_info.get("completed") is not None:
                        break
            await asyncio.sleep(delay_s)
        raise RuntimeError("OpenCode task completed without a text result.")

    @staticmethod
    def extract_text(payload: dict[str, Any]) -> str:
        data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        parts = data.get("parts") if isinstance(data, dict) else None
        texts = OpencodeClient._extract_text_parts(parts)
        if texts:
            return "\n".join(part for part in texts if part).strip()
        if isinstance(data, dict):
            info = data.get("info") or {}
            title = info.get("title") if isinstance(info, dict) else None
            if isinstance(title, str) and title.strip():
                return title.strip()
        return ""

    @staticmethod
    def _extract_text_parts(parts: Any) -> list[str]:
        texts: list[str] = []
        if not isinstance(parts, list):
            return texts
        for part in parts:
            if isinstance(part, str):
                if part.strip():
                    texts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue
            if (
                part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and part["text"].strip()
            ):
                texts.append(part["text"].strip())
            nested = part.get("parts")
            if isinstance(nested, list):
                texts.extend(OpencodeClient._extract_text_parts(nested))
        return texts

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._client.request(method, path, **kwargs)
        if response.is_error:
            raise OpencodeServeError(
                f"OpenCode Serve request failed ({response.status_code}): {response.text[:500]}"
            )
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise OpencodeServeError(
                f"OpenCode Serve returned invalid JSON: {response.text[:500]}"
            ) from exc
        return data if isinstance(data, dict) else {"data": data}
