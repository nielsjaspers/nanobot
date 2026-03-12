from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import OpenCodeServeConfig
from nanobot.opencode_client import OpencodeServeClient


@pytest.mark.asyncio
async def test_inject_context_returns_message_when_no_running_task(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    result = await mgr.inject_context("telegram:123", "more info")

    assert result == "No running OpenCode task found for this chat."


@pytest.mark.asyncio
async def test_spawn_uses_explicit_opencode_backend(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    config = OpenCodeServeConfig(enabled=True)
    mgr = SubagentManager(
        provider=provider, workspace=tmp_path, bus=MessageBus(), opencode_config=config
    )

    started = asyncio.Event()

    async def fake_run(task_id: str, task: str, label: str, origin: dict[str, str]) -> None:
        started.set()

    monkeypatch.setattr(mgr, "_run_opencode_task", fake_run)

    result = await mgr.spawn("investigate this", use_opencode=True)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert "via opencode runtime" in result


@pytest.mark.asyncio
async def test_poll_opencode_tasks_returns_completed_messages(tmp_path, monkeypatch) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    async def fake_run(task_id: str, task: str, label: str, origin: dict[str, str]) -> None:
        return None

    monkeypatch.setattr(mgr, "_run_opencode_task", fake_run)

    task_result = await mgr.spawn("do thing", use_opencode=True)
    task_id = task_result.split("id: ")[-1].split(")", 1)[0]
    info = mgr._running_tasks[task_id]
    info.session_id = "sess-1"

    async def fake_list_messages(self, session_id: str):
        assert session_id == "sess-1"
        return [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "done"}]}]

    monkeypatch.setattr(OpencodeServeClient, "list_messages", fake_list_messages)

    completed = await mgr.poll_opencode_tasks()
    await asyncio.sleep(0)

    assert len(completed) == 1
    assert completed[0][0].task_id == task_id
    assert completed[0][2] == "done"
