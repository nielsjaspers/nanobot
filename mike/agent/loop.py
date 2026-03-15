"""Simplified native tool-calling loop for Mike."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from mike.bus import MessageBus
from mike.memory.archive import ArchiveManager
from mike.chat.models import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    clamp_max_tokens,
    model_supports_vision,
)
from mike.chat.prompts import build_system_prompt
from mike.chat.reasoning import build_reasoning_kwargs
from mike.config import MikeConfig
from mike.skills import build_summary
from mike.storage.chats import ChatSession, ChatStore
from mike.tasks.research import ResearchManager
from mike.tools.delegate import OpenCodeDelegateTool
from mike.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from mike.tools.history import GetHistoryConversationTool, SearchHistoryTool
from mike.tools.memory import ReadMemoryTool
from mike.tools.message import MessageTool
from mike.tools.registry import ToolRegistry
from mike.tools.research import ResearchTool
from mike.tools.shell import ExecTool
from mike.tools.web import WebFetchTool, WebSearchTool
from mike.types import InboundMessage, OutboundMessage
from mike.helpers import build_assistant_message, detect_image_mime
from mike.llm import LLMProvider


class ContextBuilder:
    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"

    def __init__(self, store: ChatStore):
        self.store = store

    def build_system_prompt(self, session_key: str) -> str:
        root = self.store.shared_root
        return build_system_prompt(root, skills_summary=build_summary(root))

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        lines = [f"Current Time: {now}"]
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def build_messages(
        self,
        session_key: str,
        history: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        runtime = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)
        if isinstance(user_content, str):
            merged = f"{runtime}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime}] + user_content
        return [
            {"role": "system", "content": self.build_system_prompt(session_key)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        if not media:
            return text
        images = []
        attachments = []
        for path in media:
            file_path = Path(path)
            if not file_path.is_file():
                continue
            raw = file_path.read_bytes()
            mime = detect_image_mime(raw)
            if mime:
                import base64

                images.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"
                        },
                    }
                )
            else:
                attachments.append(str(file_path))
        parts: list[dict[str, Any]] = []
        if attachments:
            parts.append(
                {
                    "type": "text",
                    "text": "Attached files:\n" + "\n".join(f"- {item}" for item in attachments),
                }
            )
        parts.extend(images)
        parts.append({"type": "text", "text": text})
        return parts

    def add_tool_result(
        self, messages: list[dict[str, Any]], tool_call_id: str, tool_name: str, result: str
    ) -> list[dict[str, Any]]:
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result}
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        messages.append(
            build_assistant_message(
                content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )
        return messages


class AgentLoop:
    _TOOL_RESULT_MAX_CHARS = 16_000
    _MODEL_ALIASES = {
        "minimax": "minimax-m2.5",
        "kimi": "kimi-k2.5",
        "glm": "glm-5",
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        config: MikeConfig,
        store: ChatStore,
        research: ResearchManager,
    ):
        self.bus = bus
        self.provider = provider
        self.config = config
        self.store = store
        self.research = research
        self.model = config.default_model or DEFAULT_MODEL
        self.context = ContextBuilder(store)
        self.archiver = ArchiveManager(store, provider, self._get_effective_model)
        self.tools = ToolRegistry()
        self._running = False
        self._processing_lock = asyncio.Lock()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        root = self.config.project_root_path
        allowed = root if self.config.restrict_shell_to_project else None
        self.tools.register(ReadFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(WriteFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(EditFileTool(workspace=root, allowed_dir=allowed))
        self.tools.register(ListDirTool(workspace=root, allowed_dir=allowed))
        self.tools.register(
            ExecTool(
                timeout=self.config.command_timeout,
                working_dir=str(root),
                restrict_to_workspace=self.config.restrict_shell_to_project,
            )
        )
        self.tools.register(
            WebSearchTool(
                cli_bin=self.config.opencode_server_bin,
                attach_url=self.config.opencode_server_url,
                provider_id=self.config.opencode_model_provider_id,
            )
        )
        self.tools.register(WebFetchTool(proxy=self.config.telegram_proxy))
        self.tools.register(ReadMemoryTool(self.store.memory_path))
        self.tools.register(SearchHistoryTool(self.store.history_index_path))
        self.tools.register(GetHistoryConversationTool(self.store.history_record_path))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(OpenCodeDelegateTool(manager=self.research))
        self.tools.register(ResearchTool(manager=self.research))

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
                continue
            if cmd == "/restart":
                await self._handle_restart(msg)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(
                lambda t, key=msg.session_key: self._active_tasks.get(key, [])
                and self._active_tasks[key].remove(t)
                if t in self._active_tasks.get(key, [])
                else None
            )

    def stop(self) -> None:
        self._running = False

    async def _handle_stop(self, msg: InboundMessage) -> None:
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for task in tasks if not task.done() and task.cancel())
        for task in tasks:
            try:
                await task
            except Exception:
                pass
        research_cancelled = await self.research.cancel_by_session(msg.session_key)
        total = cancelled + research_cancelled
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Stopped {total} task(s)." if total else "No active task to stop.",
            )
        )

    async def _handle_restart(self, msg: InboundMessage) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")
        )

        async def do_restart() -> None:
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    )
                )

    def _set_tool_context(
        self, channel: str, chat_id: str, message_id: str | None = None, model: str | None = None
    ) -> None:
        for name in ("message", "web_search", "opencode_delegate", "research"):
            tool = self.tools.get(name)
            setter = getattr(tool, "set_context", None)
            if not callable(setter):
                continue
            if name == "message":
                setter(channel, chat_id, message_id)
            else:
                setter(channel, chat_id, model)

    def _get_effective_model(self, session: ChatSession) -> str:
        if session.current_model and session.current_model in SUPPORTED_MODELS:
            return session.current_model
        return self.model

    def _has_vision_content(self, msg: InboundMessage) -> bool:
        for item in msg.media:
            lower = item.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                return True
        return False

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        def fmt(call: Any) -> str:
            args = call.arguments or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) and args else None
            if not isinstance(val, str):
                return call.name
            return f'{call.name}("{val[:40]}...")' if len(val) > 40 else f'{call.name}("{val}")'

        return ", ".join(fmt(call) for call in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        model: str | None = None,
    ) -> tuple[str | None, list[str], list[dict[str, Any]]]:
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        effective_model = model or self.model or DEFAULT_MODEL
        reasoning = build_reasoning_kwargs(effective_model)
        max_tokens = clamp_max_tokens(effective_model, self.config.max_tokens)
        while iteration < self.config.max_tool_iterations:
            iteration += 1
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=effective_model,
                max_tokens=max_tokens,
                thinking=reasoning.get("thinking"),
                reasoning_effort=reasoning.get("reasoning_effort"),
            )
            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought and self.config.send_progress:
                        await on_progress(thought)
                    if self.config.send_tool_hints:
                        await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)
                tool_call_dicts = [call.to_openai_tool_call() for call in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                continue
            clean = self._strip_think(response.content)
            if response.finish_reason == "error":
                final_content = clean or "Sorry, I encountered an error calling the AI model."
                break
            messages = self.context.add_assistant_message(
                messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            final_content = clean
            break
        if final_content is None and iteration >= self.config.max_tool_iterations:
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.config.max_tool_iterations}) "
                "without completing the task."
            )
        return final_content, tools_used, messages

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        if msg.channel == "system":
            if (msg.metadata or {}).get("_task_result"):
                session = self.store.get(msg.chat_id)
                session.add_message(
                    "system",
                    msg.content,
                    task_id=(msg.metadata or {}).get("task_id"),
                    task_label=(msg.metadata or {}).get("task_label"),
                    task_status=(msg.metadata or {}).get("task_status"),
                )
                self.store.save(session)
                return None
        key = session_key or msg.session_key
        session = self.store.get(key)
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            if not session.has_meaningful_content():
                session.clear()
                self.store.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content="New session started."
                )
            try:
                archived = await self.archiver.archive_session(
                    session, channel=msg.channel, chat_id=msg.chat_id
                )
            except Exception as exc:
                logger.exception("Failed to archive session {}", key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Archiving failed, session not cleared. {exc}",
                )
            session = self.store.reset(key, preserve_model=True)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Conversation archived as '{archived.title}' and new session started.",
            )
        if cmd == "/clear":
            session = self.store.reset(key, preserve_model=True)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="Chat cleared."
            )
        if cmd == "/help":
            lines = [
                "Mike commands:",
                "/new - Start a new conversation",
                "/clear - Clear chat instantly",
                "/stop - Stop the current task",
                "/restart - Restart the bot",
                "/help - Show available commands",
                "/model - Show current model and available options",
                "/model <name> - Switch to a different model",
                "/model reset - Reset to default model",
                "/research <task> - Run a background research task",
                "/status - Show running background tasks",
                "/context <text> - Add context to a running task",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )
        if cmd == "/model" or cmd.startswith("/model "):
            return await self._handle_model_command(msg, session)
        if cmd.startswith("/research"):
            task = msg.content[len("/research") :].strip()
            if not task:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /research <task description>",
                )
            result = await self.research.start_task(
                query=task,
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                model=self._get_effective_model(session),
                kind="research",
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)
        if cmd == "/status":
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=self.research.format_status(key)
            )
        if cmd.startswith("/context"):
            extra = msg.content[len("/context") :].strip()
            if not extra:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Usage: /context <extra information>",
                )
            result = await self.research.inject_context(key, extra)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        effective_model = self._get_effective_model(session)
        if self._has_vision_content(msg) and not model_supports_vision(effective_model):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model {effective_model} does not support images. Switch to {DEFAULT_MODEL} with /model {DEFAULT_MODEL}.",
            )

        self._set_tool_context(
            msg.channel, msg.chat_id, msg.metadata.get("message_id"), effective_model
        )
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.start_turn()
        history = session.history()
        initial_messages = self.context.build_messages(
            key,
            history,
            msg.content,
            media=msg.media or None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta
                )
            )

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or bus_progress,
            model=effective_model,
        )
        final_content = final_content or "I've completed processing but have no response to give."
        self._save_turn(session, all_msgs, 1 + len(history))
        self.store.save(session)
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return None
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
        )

    async def _handle_model_command(
        self, msg: InboundMessage, session: ChatSession
    ) -> OutboundMessage:
        parts = msg.content.strip().split(maxsplit=1)
        if len(parts) == 1:
            current = self._get_effective_model(session)
            lines = [f"Current model: {current}", "", "Available models:"]
            for name, info in SUPPORTED_MODELS.items():
                vision_badge = " [vision]" if info["vision"] else ""
                marker = "-> " if name == current else "   "
                lines.append(f"{marker}{name}{vision_badge} - {info['description']}")
            lines.append("")
            lines.append("Usage: /model <name> to switch, /model reset to restore default")
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines)
            )
        subcmd = parts[1].lower().strip()
        subcmd = self._MODEL_ALIASES.get(subcmd, subcmd)
        if subcmd == "reset":
            old = self._get_effective_model(session)
            session.current_model = None
            self.store.save(session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model reset from {old} to default ({self._get_effective_model(session)}).",
            )
        if subcmd in SUPPORTED_MODELS:
            old = self._get_effective_model(session)
            session.current_model = subcmd
            self.store.save(session)
            note = " Supports images." if SUPPORTED_MODELS[subcmd]["vision"] else " Text-only."
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Model switched from {old} to {subcmd}.{note}",
            )
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"Unknown model: '{subcmd}'. Available: {', '.join(SUPPORTED_MODELS)}",
        )

    def _save_turn(self, session: ChatSession, messages: list[dict[str, Any]], skip: int) -> None:
        for message in messages[skip:]:
            entry = dict(message)
            role = entry.get("role")
            content = entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            if (
                role == "tool"
                and isinstance(content, str)
                and len(content) > self._TOOL_RESULT_MAX_CHARS
            ):
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            if role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for part in content:
                        if (
                            part.get("type") == "text"
                            and isinstance(part.get("text"), str)
                            and part["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                        ):
                            continue
                        if part.get("type") == "image_url" and part.get("image_url", {}).get(
                            "url", ""
                        ).startswith("data:image/"):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(part)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now().isoformat()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            session_key_override=session_key,
        )
        response = await self._process_message(
            msg, session_key=session_key, on_progress=on_progress
        )
        return response.content if response else ""
