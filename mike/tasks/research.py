"""Research and delegation orchestration for Mike."""

from __future__ import annotations

import asyncio
import uuid

from loguru import logger

from mike.bus import MessageBus
from mike.chat.reasoning import build_reasoning_kwargs
from mike.config import MikeConfig
from mike.opencode.client import OpencodeClient
from mike.opencode.runner import OpencodeRunner
from mike.storage.tasks import TaskRecord, TaskStore
from mike.tasks.manager import TaskManager
from mike.types import InboundMessage, OutboundMessage


def build_opencode_reasoning_config(model: str) -> dict[str, str | int] | None:
    policy = build_reasoning_kwargs(model)
    if policy.get("thinking"):
        return {"type": "thinking"}
    effort = policy.get("reasoning_effort")
    if not effort:
        return None
    return {"type": "reasoning_effort", "value": str(effort)}


class ResearchManager:
    def __init__(self, config: MikeConfig, bus: MessageBus, store: TaskStore, manager: TaskManager):
        self.config = config
        self.bus = bus
        self.store = store
        self.manager = manager

    async def start_task(
        self,
        query: str,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        title: str = "research",
        model: str | None = None,
        kind: str = "research",
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = TaskRecord(
            task_id=task_id,
            session_key=session_key,
            origin_channel=channel,
            origin_chat_id=chat_id,
            kind=kind,
            query=query,
            title=title,
            model=model or self.config.default_model,
            phase="QUEUED",
            status="queued",
        )
        self.store.save(task)
        self.store.append_event(
            task_id, "created", {"query": query, "model": task.model, "kind": kind}
        )
        raw = asyncio.create_task(self._run_task(task_id))
        self.manager.add(task_id, session_key, raw)
        return f"{kind.title()} task [{task_id}] started via opencode runtime. Use /status or /context."

    async def _run_task(self, task_id: str) -> None:
        task = self.store.load(task_id)
        if task is None:
            return
        client = OpencodeClient(
            base_url=self.config.opencode_server_url,
            username="opencode",
            password=self.config.opencode_server_password,
            directory=str(self.config.project_root_path),
        )
        try:
            session = await client.create_session(title=task.title)
            session_id = session.get("id") or session.get("data", {}).get("id")
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError("OpenCode Serve did not return a session id")
            task.backend_session_id = session_id
            task.phase = "RUNNING"
            task.status = "running"
            self.store.save(task)
            self.store.append_event(task_id, "session_started", {"session_id": session_id})
            system = self._build_system_prompt(task)
            await client.prompt_async(
                session_id=session_id,
                text=f"Task:\n{task.query}",
                system=system,
                provider_id=self.config.opencode_model_provider_id,
                model_id=task.model,
                agent=self.config.opencode_agent,
                reasoning_config=build_opencode_reasoning_config(
                    task.model or self.config.default_model
                ),
            )
            await self._poll_until_done(task, client)
        except asyncio.CancelledError:
            current = self.store.load(task_id)
            if current and current.backend_session_id:
                try:
                    await client.abort_session(current.backend_session_id)
                except Exception:
                    logger.debug("Task {} abort failed", task_id)
            current = self.store.load(task_id)
            if current:
                current.phase = "CANCELLED"
                current.status = "cancelled"
                self.store.save(current)
                self.store.append_event(task_id, "cancelled", {})
            raise
        except Exception as exc:
            logger.exception("Task {} failed", task_id)
            current = self.store.load(task_id)
            if current:
                current.phase = "FAILED"
                current.status = "failed"
                current.error = str(exc)
                self.store.save(current)
                self.store.append_event(task_id, "failed", {"error": str(exc)})
                await self._announce_failure(current)
        finally:
            await client.aclose()

    async def _poll_until_done(self, task: TaskRecord, client: OpencodeClient) -> None:
        assert task.backend_session_id
        last = ""
        while True:
            items = await client.list_messages(task.backend_session_id)
            text = self._latest_assistant_text(items)
            if text and text != last:
                last = text
                task.progress_summary = self._compact(text)
                self.store.save(task)
                self.store.append_event(
                    task.task_id, "progress", {"summary": task.progress_summary}
                )
                await self._publish_progress(task, task.progress_summary)
            if self._is_finished(items):
                break
            await asyncio.sleep(2)
        final = await client.wait_for_text(task.backend_session_id)
        task.final_summary = final
        task.phase = "COMPLETE"
        task.status = "completed"
        task.final_artifact = self.store.write_artifact(task.task_id, "report.md", final)
        self.store.save(task)
        self.store.append_event(task.task_id, "completed", {"artifact": task.final_artifact})
        await self._announce_completion(task)

    async def run_delegated_once(self, prompt: str, *, model: str | None = None) -> str:
        runner = OpencodeRunner(self.config.opencode_server_bin, self.config.opencode_server_url)
        out, err, code = await runner.run(
            prompt, model=model, provider_id=self.config.opencode_model_provider_id
        )
        if code not in (0, None):
            return err.strip() or out.strip() or "OpenCode delegation failed"
        return out.strip() or "Delegated task completed."

    async def inject_context(self, session_key: str, text: str, task_id: str | None = None) -> str:
        task = self._find_task(session_key, task_id)
        if task is None:
            return "No running research task found for this chat."
        task.user_injections.append(text)
        self.store.save(task)
        self.store.append_injection(task.task_id, text)
        if not task.backend_session_id:
            return f"Context queued for task {task.task_id}."
        client = OpencodeClient(
            base_url=self.config.opencode_server_url,
            username="opencode",
            password=self.config.opencode_server_password,
            directory=str(self.config.project_root_path),
        )
        try:
            await client.prompt(
                session_id=task.backend_session_id,
                text=f"Additional context from the user:\n\n{text}",
                provider_id=self.config.opencode_model_provider_id,
                model_id=task.model,
                agent=self.config.opencode_agent,
                no_reply=True,
                reasoning_config=build_opencode_reasoning_config(
                    task.model or self.config.default_model
                ),
            )
        finally:
            await client.aclose()
        return f"Context added to task {task.task_id}."

    async def cancel_by_session(self, session_key: str) -> int:
        count = 0
        for task in self.store.list():
            if task.session_key != session_key:
                continue
            if task.status not in {"queued", "running"}:
                continue
            await self.cancel_task(task.task_id)
            count += 1
        return count

    async def cancel_task(self, task_id: str) -> None:
        task = self.store.load(task_id)
        if task is None:
            return
        if task.backend_session_id:
            client = OpencodeClient(
                base_url=self.config.opencode_server_url,
                username="opencode",
                password=self.config.opencode_server_password,
                directory=str(self.config.project_root_path),
            )
            try:
                await client.abort_session(task.backend_session_id)
            finally:
                await client.aclose()
        await self.manager.cancel_task(task_id)
        task.phase = "CANCELLED"
        task.status = "cancelled"
        self.store.save(task)

    def format_status(self, session_key: str | None = None) -> str:
        tasks = self.store.list()
        if session_key is not None:
            tasks = [task for task in tasks if task.session_key == session_key]
        if not tasks:
            return "No background tasks found for this chat."
        lines = ["Background tasks:"]
        for task in tasks[:10]:
            extra = f" summary={task.progress_summary}" if task.progress_summary else ""
            lines.append(f"- {task.task_id} [{task.kind}] {task.status} phase={task.phase}{extra}")
        return "\n".join(lines)

    async def _publish_progress(self, task: TaskRecord, summary: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=task.origin_channel,
                chat_id=task.origin_chat_id,
                content=f"[{task.task_id} running] {summary}",
                metadata={"_progress": True, "_research_task": True, "task_id": task.task_id},
            )
        )

    async def _announce_completion(self, task: TaskRecord) -> None:
        final = task.final_summary or "Task completed."
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
                    f"Task result for '{task.title}' (completed).\n\n"
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

    async def _announce_failure(self, task: TaskRecord) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=task.origin_channel,
                chat_id=task.origin_chat_id,
                content=f"[{task.task_id} failed]\n\n{task.error or 'Task failed.'}",
                metadata={"_task_result": True, "_research_task": True, "task_id": task.task_id},
            )
        )

    def _build_system_prompt(self, task: TaskRecord) -> str:
        if task.kind != "research":
            return "You are running a delegated Mike task inside OpenCode. Finish the task cleanly."
        return (
            "You are running a delegated Mike research task inside OpenCode. "
            "Follow the deep-research methodology in the loaded research skill carefully."
        )

    def _find_task(self, session_key: str, task_id: str | None) -> TaskRecord | None:
        tasks = [
            task
            for task in self.store.list()
            if task.session_key == session_key and task.status in {"queued", "running"}
        ]
        if task_id:
            for task in tasks:
                if task.task_id == task_id:
                    return task
            return None
        return tasks[0] if tasks else None

    @staticmethod
    def _latest_assistant_text(items: list[dict]) -> str:
        text = ""
        for item in items:
            if not isinstance(item, dict):
                continue
            info = item.get("info") or {}
            if info.get("role") != "assistant":
                continue
            candidate = OpencodeClient.extract_text(item)
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
