"""Subagent manager for background task execution."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig, OpenCodeServeConfig
from nanobot.opencode_client import OpencodeServeClient
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_assistant_message

from loguru import logger


@dataclass
class RunningTaskInfo:
    """Bookkeeping for a background task."""

    task_id: str
    label: str
    backend: str
    raw_task: asyncio.Task[None]
    session_id: str | None = None
    session_key: str | None = None
    status: str = "running"


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        opencode_config: OpenCodeServeConfig | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.opencode_config = opencode_config or OpenCodeServeConfig()
        self._running_tasks: dict[str, RunningTaskInfo] = {}
        self._session_tasks: dict[str, set[str]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        use_opencode: bool | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}
        backend = self._select_backend(use_opencode, task)

        runner = (
            self._run_opencode_task(task_id, task, display_label, origin)
            if backend == "opencode"
            else self._run_native_subagent(task_id, task, display_label, origin)
        )
        bg_task = asyncio.create_task(runner)
        info = RunningTaskInfo(
            task_id=task_id, label=display_label, backend=backend, raw_task=bg_task
        )
        info.session_key = session_key
        self._running_tasks[task_id] = info
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}] via {}: {}", task_id, backend, display_label)
        return (
            f"Task [{display_label}] started via {backend} runtime (id: {task_id}). "
            "I'll notify you when it completes."
        )

    def _select_backend(self, use_opencode: bool | None, task: str) -> str:
        if use_opencode is not None:
            return "opencode" if use_opencode else "native"
        return "native"

    async def _run_native_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting native task: {}", task_id, label)

        try:
            tools = self._build_native_tools()
            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                    messages.append(
                        build_assistant_message(
                            response.content or "",
                            tool_calls=tool_call_dicts,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        )
                    )

                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug(
                            "Subagent [{}] executing: {} with arguments: {}",
                            task_id,
                            tool_call.name,
                            args_str,
                        )
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            }
                        )
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            self._running_tasks[task_id].status = "completed"
            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            self._running_tasks[task_id].status = "failed"
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _run_opencode_task(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the task through OpenCode Serve and announce the result."""
        logger.info("Subagent [{}] starting OpenCode task: {}", task_id, label)
        client = OpencodeServeClient(
            base_url=self.opencode_config.url,
            username=self.opencode_config.username,
            password=self.opencode_config.password,
        )
        try:
            session = await client.create_session(title=label)
            session_id = session.get("id") or session.get("data", {}).get("id")
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("OpenCode Serve did not return a session id")
            self._running_tasks[task_id].session_id = session_id
            prompt = self._build_opencode_prompt(task)
            await client.prompt_async(
                session_id=session_id,
                text=prompt,
                provider_id=self.opencode_config.model_provider_id,
                model_id=self.opencode_config.model_id,
                agent=self.opencode_config.agent,
            )
            logger.info("Subagent [{}] OpenCode session started: {}", task_id, session_id)
        except Exception as e:
            self._running_tasks[task_id].status = "failed"
            logger.error("Subagent [{}] OpenCode execution failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, f"Error: {e}", origin, "error")
        finally:
            await client.aclose()

    def _build_native_tools(self) -> ToolRegistry:
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            )
        )
        tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"
        task_info = self._running_tasks.get(task_id)
        backend = "task"
        if task_info is not None:
            backend = task_info.backend

        announce_content = f"""[{backend} task '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like task IDs unless relevant."""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            "Subagent [{}] announced result to {}:{}", task_id, origin["channel"], origin["chat_id"]
        )

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [
            f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""
        ]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(
                f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}"
            )

        return "\n\n".join(parts)

    def _build_opencode_prompt(self, task: str) -> str:
        return (
            "You are handling a delegated Nanobot task. Stay focused, use tools when useful, "
            "and produce a concise final result that can be forwarded back to the user.\n\n"
            f"Task:\n{task}"
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        infos = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].raw_task.done()
        ]
        cancel_count = len(infos)
        abort_coros = []
        for info in infos:
            info.raw_task.cancel()
            if info.backend == "opencode" and info.session_id:
                client = OpencodeServeClient(
                    base_url=self.opencode_config.url,
                    username=self.opencode_config.username,
                    password=self.opencode_config.password,
                )
                abort_coros.append(self._abort_remote_session(client, info.session_id))
        if infos:
            await asyncio.gather(*(info.raw_task for info in infos), return_exceptions=True)
        if abort_coros:
            await asyncio.gather(*abort_coros, return_exceptions=True)
        return cancel_count

    async def _abort_remote_session(self, client: OpencodeServeClient, session_id: str) -> None:
        try:
            await client.abort_session(session_id)
        finally:
            await client.aclose()

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return sum(1 for info in self._running_tasks.values() if not info.raw_task.done())

    def get_tasks(self, session_key: str | None = None) -> list[RunningTaskInfo]:
        """Return task info, optionally scoped to a session."""
        tasks = list(self._running_tasks.values())
        if session_key is None:
            return tasks
        return [task for task in tasks if task.session_key == session_key]

    async def poll_opencode_tasks(self) -> list[tuple[RunningTaskInfo, str, str]]:
        """Poll running OpenCode tasks and return completed ones with final text."""
        done: list[tuple[RunningTaskInfo, str, str]] = []
        pending = [
            task
            for task in self._running_tasks.values()
            if task.backend == "opencode" and task.status == "running" and task.session_id
        ]
        for task in pending:
            assert task.session_id is not None
            client = OpencodeServeClient(
                base_url=self.opencode_config.url,
                username=self.opencode_config.username,
                password=self.opencode_config.password,
            )
            try:
                items = await client.list_messages(task.session_id)
            except Exception:
                await client.aclose()
                continue
            finally:
                await client.aclose()
            if not isinstance(items, list) or not items:
                continue
            last = items[-1]
            if not isinstance(last, dict):
                continue
            info = last.get("info") or {}
            role = info.get("role")
            if role != "assistant":
                continue
            text = OpencodeServeClient.extract_text(last)
            if not text:
                continue
            task.status = "completed"
            done.append((task, task.label, text))
        return done

    async def inject_context(
        self,
        session_key: str,
        text: str,
        task_id: str | None = None,
    ) -> str:
        """Inject extra context into a running OpenCode-backed task."""
        candidates = [
            task
            for task in self.get_tasks(session_key)
            if task.backend == "opencode" and task.status == "running" and task.session_id
        ]
        if task_id:
            candidates = [task for task in candidates if task.task_id == task_id]
        if not candidates:
            return "No running OpenCode task found for this chat."

        target = candidates[-1]
        session_id = target.session_id
        if session_id is None:
            return "OpenCode task has no session id available."
        client = OpencodeServeClient(
            base_url=self.opencode_config.url,
            username=self.opencode_config.username,
            password=self.opencode_config.password,
        )
        try:
            await client.prompt(
                session_id=session_id,
                text=f"Additional context from the user:\n\n{text}",
                provider_id=self.opencode_config.model_provider_id,
                model_id=self.opencode_config.model_id,
                agent=self.opencode_config.agent,
                no_reply=True,
            )
        finally:
            await client.aclose()

        return f"Context added to task {target.task_id}."
