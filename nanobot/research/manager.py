"""Manager for OpenCode-backed research tasks."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nanobot.models_config import get_model_reasoning_config
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.opencode_client import OpencodeServeClient
from nanobot.research.models import ResearchTask
from nanobot.research.persistence import ResearchPersistence


@dataclass
class RunningResearch:
    task_id: str
    raw_task: asyncio.Task[None]
    session_key: str


class ResearchManager:
    """Nanobot-side orchestration for OpenCode-executed research."""

    def __init__(
        self,
        *,
        workspace: Path,
        provider,
        bus: MessageBus,
        tools,
        model: str,
        config,
        opencode_config,
    ):
        del provider, tools
        self.workspace = workspace
        self.bus = bus
        self.model = model
        self.config = config
        self.opencode_config = opencode_config
        self.persistence = ResearchPersistence(workspace)
        self._running: dict[str, RunningResearch] = {}
        self._research_skill_content = self._load_research_skill()

    def _load_research_skill(self) -> str | None:
        """Load the deep-research skill content directly from builtin skills."""
        import re
        from pathlib import Path

        # Load directly from builtin skills directory
        builtin_skills_dir = Path(__file__).parent.parent / "skills"
        skill_path = builtin_skills_dir / "deep-research" / "SKILL.md"

        if not skill_path.exists():
            return None

        content = skill_path.read_text(encoding="utf-8")

        # Strip frontmatter
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                content = content[match.end():].strip()

        return content

    def bind_tools(self, tools) -> None:
        del tools

    def resolve_model(self, model: str | None) -> str:
        return model or self.opencode_config.model_id or self.model

    async def start_task(
        self,
        query: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        backend: str | None = None,
        title: str = "research",
        model: str | None = None,
    ) -> str:
        del backend
        if not self.config.enabled:
            return "Error: Research tasks are disabled in config."
        if not self.opencode_config.enabled:
            return "Error: OpenCode Serve is not enabled in config."

        task_id = str(uuid.uuid4())[:8]
        task = ResearchTask(
            task_id=task_id,
            session_key=session_key,
            origin_channel=channel,
            origin_chat_id=chat_id,
            query=query,
            backend="opencode",
            title=title,
            model=self.resolve_model(model),
            phase="QUEUED",
            status="queued",
        )
        self.persistence.save_task(task)
        self.persistence.append_event(task_id, "created", {"query": query, "model": task.model})
        raw = asyncio.create_task(self._run_task(task_id))
        self._running[task_id] = RunningResearch(
            task_id=task_id, raw_task=raw, session_key=session_key
        )
        raw.add_done_callback(lambda _: self._running.pop(task_id, None))
        return (
            f"Research task [{task_id}] started via opencode runtime. "
            "Use /status to inspect progress or /context to add more information."
        )

    async def run_local_task(
        self,
        *,
        task: str,
        label: str | None,
        session_key: str,
        channel: str,
        chat_id: str,
        model: str | None = None,
    ) -> str:
        title = label or task[:40].strip() or "task"
        return await self.start_task(
            query=task,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            title=title,
            model=model,
        )

    async def _run_task(self, task_id: str) -> None:
        task = self.persistence.load_task(task_id)
        if task is None:
            return
        client = self._client(task.model)
        try:
            session = await client.create_session(title=task.title)
            session_id = session.get("id") or session.get("data", {}).get("id")
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("OpenCode Serve did not return a session id")
            task.backend_session_id = session_id
            task.phase = "RUNNING"
            task.status = "running"
            self.persistence.save_task(task)
            self.persistence.append_event(task_id, "session_started", {"session_id": session_id})
            await client.prompt_async(
                session_id=session_id,
                text=self._build_prompt(task),
                system=self._build_system_prompt(),
                provider_id=self.opencode_config.model_provider_id,
                model_id=task.model,
                agent=self.config.opencode_agent or self.opencode_config.agent,
                reasoning_config=get_model_reasoning_config(task.model) if task.model else None,
            )
            await self._poll_until_done(task, client)
        except asyncio.CancelledError:
            current = self.persistence.load_task(task_id)
            if current and current.backend_session_id:
                try:
                    await client.abort_session(current.backend_session_id)
                except Exception:
                    logger.debug("Research task {} abort failed", task_id)
            current = self.persistence.load_task(task_id)
            if current:
                current.phase = "CANCELLED"
                current.status = "cancelled"
                self.persistence.save_task(current)
                self.persistence.append_event(task_id, "cancelled", {})
            raise
        except Exception as exc:
            logger.exception("Research task {} failed", task_id)
            current = self.persistence.load_task(task_id)
            if current:
                current.phase = "FAILED"
                current.status = "failed"
                current.error = str(exc)
                self.persistence.save_task(current)
                self.persistence.append_event(task_id, "failed", {"error": str(exc)})
                await self._announce_failure(current)
        finally:
            await client.aclose()

    async def _poll_until_done(self, task: ResearchTask, client: OpencodeServeClient) -> None:
        assert task.backend_session_id
        last = ""
        while True:
            items = await client.list_messages(task.backend_session_id)
            text = self._latest_assistant_text(items)
            if text and text != last:
                last = text
                task.progress_summary = self._compact(text)
                self.persistence.save_task(task)
                self.persistence.append_event(
                    task.task_id, "progress", {"summary": task.progress_summary}
                )
                await self._publish_progress(task, task.progress_summary)
            if self._is_finished(items):
                break
            await asyncio.sleep(self.config.progress_poll_seconds)

        final = await client.wait_for_text(task.backend_session_id)
        task.final_summary = final
        task.phase = "COMPLETE"
        task.status = "completed"
        task.final_artifact = self.persistence.write_artifact(task.task_id, "report.md", final)
        self.persistence.save_task(task)
        self.persistence.append_event(task.task_id, "completed", {"artifact": task.final_artifact})
        await self._announce_completion(task)

    async def _publish_progress(self, task: ResearchTask, summary: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=task.origin_channel,
                chat_id=task.origin_chat_id,
                content=f"[{task.task_id} running] {summary}",
                metadata={"_progress": True, "_research_task": True, "task_id": task.task_id},
            )
        )

    async def _announce_completion(self, task: ResearchTask) -> None:
        final = task.final_summary or "Research task completed."
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=task.origin_channel,
                chat_id=task.origin_chat_id,
                content=f"[{task.task_id} completed]\n\n{final}",
                metadata={"_task_result": True, "_research_task": True, "task_id": task.task_id},
            )
        )
        await self.bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id="research",
                chat_id=task.session_key,
                content=(
                    f"Research task result for '{task.title}' (completed).\n\n"
                    f"Task ID: {task.task_id}\n\n"
                    f"Original task:\n{task.query}\n\n"
                    f"Actual result:\n{final}"
                ),
                metadata={
                    "_task_result": True,
                    "_research_task": True,
                    "task_id": task.task_id,
                    "task_label": task.title,
                    "task_status": "completed",
                },
            )
        )

    async def _announce_failure(self, task: ResearchTask) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=task.origin_channel,
                chat_id=task.origin_chat_id,
                content=f"[{task.task_id} failed]\n\n{task.error or 'Research task failed.'}",
                metadata={"_task_result": True, "_research_task": True, "task_id": task.task_id},
            )
        )

    def format_status(self, session_key: str | None = None) -> str:
        tasks = self.persistence.list_tasks()
        if session_key is not None:
            tasks = [task for task in tasks if task.session_key == session_key]
        if not tasks:
            return "No research tasks found for this chat."
        lines = ["Research tasks:"]
        for task in tasks[:10]:
            extra = f" summary={task.progress_summary}" if task.progress_summary else ""
            lines.append(
                f"- {task.task_id} [{task.backend}] {task.status} phase={task.phase}{extra}"
            )
        return "\n".join(lines)

    async def inject_context(self, session_key: str, text: str, task_id: str | None = None) -> str:
        task = self._find_task(session_key, task_id)
        if task is None:
            return "No running research task found for this chat."
        task.user_injections.append(text)
        self.persistence.save_task(task)
        self.persistence.append_event(task.task_id, "context", {"text": text})
        if not task.backend_session_id:
            return f"Context queued for research task {task.task_id}."
        client = self._client(task.model)
        try:
            await client.prompt(
                session_id=task.backend_session_id,
                text=f"Additional context from the user:\n\n{text}",
                provider_id=self.opencode_config.model_provider_id,
                model_id=task.model,
                agent=self.config.opencode_agent or self.opencode_config.agent,
                no_reply=True,
                reasoning_config=get_model_reasoning_config(task.model) if task.model else None,
            )
        finally:
            await client.aclose()
        return f"Context added to research task {task.task_id}."

    async def cancel_by_session(self, session_key: str) -> int:
        count = 0
        for task in self.persistence.list_tasks():
            if task.session_key != session_key:
                continue
            if task.status not in {"queued", "running"}:
                continue
            await self.cancel_task(task.task_id)
            count += 1
        return count

    async def cancel_task(self, task_id: str) -> None:
        task = self.persistence.load_task(task_id)
        if task is None:
            return
        if task.backend_session_id:
            client = self._client(task.model)
            try:
                await client.abort_session(task.backend_session_id)
            finally:
                await client.aclose()
        running = self._running.get(task_id)
        if running and not running.raw_task.done():
            running.raw_task.cancel()
            await asyncio.gather(running.raw_task, return_exceptions=True)
        task.phase = "CANCELLED"
        task.status = "cancelled"
        self.persistence.save_task(task)

    async def resume_pending(self) -> None:
        if not self.config.auto_resume:
            return
        for task in self.persistence.list_tasks():
            if task.status not in {"queued", "running"}:
                continue
            if task.task_id in self._running:
                continue
            raw = asyncio.create_task(self._run_task(task.task_id))
            self._running[task.task_id] = RunningResearch(
                task_id=task.task_id,
                raw_task=raw,
                session_key=task.session_key,
            )
            raw.add_done_callback(lambda _, tid=task.task_id: self._running.pop(tid, None))

    def list_tasks(self, session_key: str | None = None) -> list[ResearchTask]:
        tasks = self.persistence.list_tasks()
        if session_key is None:
            return tasks
        return [task for task in tasks if task.session_key == session_key]

    def _find_task(self, session_key: str, task_id: str | None) -> ResearchTask | None:
        tasks = [
            task
            for task in self.persistence.list_tasks()
            if task.session_key == session_key and task.status in {"queued", "running"}
        ]
        if task_id:
            for task in tasks:
                if task.task_id == task_id:
                    return task
            return None
        return tasks[0] if tasks else None

    def _client(self, model: str | None) -> OpencodeServeClient:
        return OpencodeServeClient(
            base_url=self.opencode_config.url,
            username=self.opencode_config.username,
            password=self.opencode_config.password,
            directory=str(self.workspace),
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for research tasks."""
        base = (
            "You are running a delegated Nanobot research task inside OpenCode. "
            "Follow the deep-research methodology below carefully.\n"
        )
        if self._research_skill_content:
            base += f"\n{self._research_skill_content}\n"
        if not self.config.allow_prototypes:
            base += (
                "\nNote: Do not create or execute prototypes unless the user explicitly requested them."
            )
        return base

    def _build_prompt(self, task: ResearchTask) -> str:
        return f"Task:\n{task.query}"

    @staticmethod
    def _latest_assistant_text(items: list[dict]) -> str:
        text = ""
        for item in items:
            if not isinstance(item, dict):
                continue
            info = item.get("info") or {}
            if info.get("role") != "assistant":
                continue
            candidate = OpencodeServeClient.extract_text(item)
            if candidate:
                text = candidate
        return text

    @staticmethod
    def _is_finished(items: list[dict]) -> bool:
        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            info = item.get("info") or {}
            if info.get("role") != "assistant":
                continue
            finish = info.get("finish")
            time_info = info.get("time") if isinstance(info, dict) else None
            if finish not in (None, "tool-calls") and isinstance(time_info, dict):
                if time_info.get("completed") is not None:
                    return True
        return False

    @staticmethod
    def _compact(text: str) -> str:
        line = text.strip().replace("\n", " ")
        return line[:240] + ("..." if len(line) > 240 else "")
