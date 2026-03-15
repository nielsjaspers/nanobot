"""Direct provider for OpenAI-compatible and Anthropic-compatible APIs."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import httpx
import json_repair

from mike.chat.models import SUPPORTED_MODELS, get_model
from mike.llm import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._session_id = uuid.uuid4().hex
        self._client = httpx.AsyncClient(timeout=120.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model_id = model or self.default_model
        model_config = get_model(model_id) or {}
        api_type = model_config.get("api_type", "openai-compatible")
        if api_type == "anthropic-compatible":
            return await self._chat_anthropic(
                messages=messages,
                model_id=model_id,
                model_config=model_config,
                tools=tools,
                max_tokens=max_tokens,
                thinking=thinking,
                tool_choice=tool_choice,
            )
        return await self._chat_openai(
            messages=messages,
            model_id=model_id,
            model_config=model_config,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        model_config: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        endpoint = model_config.get("endpoint", "/chat/completions")
        auth_header = model_config.get("auth_header", "Authorization")
        auth_prefix = model_config.get("auth_prefix", "Bearer ")
        reasoning_param = model_config.get("reasoning_param", "reasoning_effort")
        reasoning_value = model_config.get("reasoning_value", "high")
        headers = {
            auth_header: f"{auth_prefix}{self.api_key}",
            "Content-Type": "application/json",
            "x-session-affinity": self._session_id,
        }
        body: dict[str, Any] = {
            "model": model_id,
            "messages": self._prepare_messages_openai(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            body[reasoning_param] = reasoning_effort
        elif reasoning_value:
            body[reasoning_param] = reasoning_value
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        url = f"{(self.api_base or 'http://localhost:8000/v1').rstrip('/')}{endpoint}"
        try:
            response = await self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_openai_response(response.json())
        except httpx.HTTPStatusError as exc:
            return LLMResponse(
                content=f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

    async def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        model_config: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking: dict[str, Any] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> LLMResponse:
        del tool_choice
        endpoint = model_config.get("endpoint", "/messages")
        auth_header = model_config.get("auth_header", "x-api-key")
        auth_prefix = model_config.get("auth_prefix", "")
        reasoning_param = model_config.get("reasoning_param", "thinking")
        reasoning_value = model_config.get("reasoning_value", {"type": "enabled"})
        headers = {
            auth_header: f"{auth_prefix}{self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-session-affinity": self._session_id,
        }
        system_prompt, chat_messages = self._prepare_messages_anthropic(messages)
        body: dict[str, Any] = {
            "model": model_id,
            "messages": chat_messages,
            "max_tokens": max(1, max_tokens),
        }
        if system_prompt:
            body["system"] = system_prompt
        if thinking:
            body[reasoning_param] = thinking
        elif reasoning_value:
            body[reasoning_param] = reasoning_value
        if tools:
            body["tools"] = self._convert_tools_anthropic(tools)
        url = f"{(self.api_base or 'http://localhost:8000/v1').rstrip('/')}{endpoint}"
        try:
            response = await self._client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_anthropic_response(response.json())
        except httpx.HTTPStatusError as exc:
            return LLMResponse(
                content=f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(content=f"Error: {exc}", finish_reason="error")

    def _parse_openai_response(self, data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="Error: No choices in response", finish_reason="error")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = [
            ToolCallRequest(
                id=self._normalize_tool_call_id(tc.get("id", "call_0")),
                name=tc.get("function", {}).get("name", ""),
                arguments=self._parse_args(tc.get("function", {}).get("arguments", "{}")),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        reasoning_content = self._extract_reasoning_content(message)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "stop",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            reasoning_content=reasoning_content,
        )

    def _parse_anthropic_response(self, data: dict[str, Any]) -> LLMResponse:
        content_blocks = data.get("content", [])
        text_content = ""
        reasoning_content = ""
        tool_calls = []
        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_content += block.get("text", "")
            elif block_type == "thinking":
                reasoning_content += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.get("id", "toolu_0"),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
        usage = data.get("usage", {})
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"
        return LLMResponse(
            content=text_content or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            reasoning_content=reasoning_content or None,
            thinking_blocks=content_blocks
            if any(b.get("type") == "thinking" for b in content_blocks)
            else None,
        )

    def _prepare_messages_openai(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = self._sanitize_empty_content(messages)
        id_map: dict[str, str] = {}
        result: list[dict[str, Any]] = []
        for msg in prepared:
            clean = dict(msg)
            if clean.get("role") == "assistant" and clean.get("tool_calls"):
                clean["tool_calls"] = [
                    self._prepare_tool_call(tc, id_map)
                    for tc in clean["tool_calls"]
                    if isinstance(tc, dict)
                ]
                if "reasoning_content" not in clean:
                    clean["reasoning_content"] = ""
            if clean.get("role") == "tool" and isinstance(clean.get("tool_call_id"), str):
                clean["tool_call_id"] = self._normalize_tool_call_id(clean["tool_call_id"])
            result.append(clean)
        return result

    def _prepare_messages_anthropic(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        system_prompt = ""
        chat_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_prompt = content
                elif isinstance(content, list):
                    texts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    system_prompt = "\n".join(texts)
                continue
            if role == "user":
                chat_messages.append(
                    {"role": "user", "content": self._convert_content_anthropic(content)}
                )
            elif role == "assistant":
                assistant_content = self._convert_content_anthropic(content)
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            assistant_content.append(
                                {
                                    "type": "tool_use",
                                    "id": self._normalize_tool_call_id(tc.get("id", "toolu_0")),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "input": self._parse_args(
                                        tc.get("function", {}).get("arguments", "{}")
                                    ),
                                }
                            )
                assistant_content = [
                    c
                    for c in assistant_content
                    if not (c.get("type") == "text" and not c.get("text"))
                ]
                if assistant_content:
                    chat_messages.append({"role": "assistant", "content": assistant_content})
            elif role == "tool":
                tool_result_content = content if isinstance(content, str) else json.dumps(content)
                chat_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": self._normalize_tool_call_id(
                                    msg.get("tool_call_id", "toolu_0")
                                ),
                                "content": tool_result_content,
                            }
                        ],
                    }
                )
        return system_prompt, chat_messages

    def _convert_content_anthropic(self, content: Any) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            result: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    result.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        result.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            if url.startswith("data:"):
                                result.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": url.split(";")[0].split(":")[1]
                                            if ":" in url
                                            else "image/png",
                                            "data": url.split(",", 1)[1] if "," in url else "",
                                        },
                                    }
                                )
                            else:
                                result.append(
                                    {"type": "image", "source": {"type": "url", "url": url}}
                                )
            return result
        return []

    def _convert_tools_anthropic(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                result.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object"}),
                    }
                )
        return result

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _prepare_tool_call(
        self, tool_call: dict[str, Any], id_map: dict[str, str]
    ) -> dict[str, Any]:
        clean = dict(tool_call)
        tool_id = clean.get("id")
        if isinstance(tool_id, str):
            clean["id"] = id_map.setdefault(tool_id, self._normalize_tool_call_id(tool_id))
        function = clean.get("function")
        if isinstance(function, dict):
            clean["function"] = dict(function)
        return clean

    @staticmethod
    def _parse_args(args: Any) -> dict[str, Any]:
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                result = json_repair.loads(args)
                if isinstance(result, dict):
                    return result
                return {"raw": args}
            except Exception:
                return {"raw": args}
        return {}

    @staticmethod
    def _extract_reasoning_content(message: dict[str, Any]) -> str | None:
        direct = message.get("reasoning_content")
        if isinstance(direct, str) and direct:
            return direct
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        details = message.get("reasoning_details")
        if isinstance(details, list):
            texts = []
            for item in details:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if isinstance(text, str) and text:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
        return None

    def get_default_model(self) -> str:
        return self.default_model
