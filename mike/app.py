"""Mike application runtime and CLI entrypoints."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from mike.agent.loop import AgentLoop
from mike.bootstrap import ensure_root
from mike.bus import MessageBus
from mike.config import MikeConfig, default_config_path, load_config, save_config
from mike.opencode.server import OpencodeServer
from mike.provider import make_provider
from mike.storage.chats import ChatStore
from mike.storage.tasks import TaskStore
from mike.tasks.manager import TaskManager
from mike.tasks.research import ResearchManager
from mike.telegram.bot import TelegramBot

app = typer.Typer(name="mike", help="Mike - personal assistant bot")
console = Console()


async def _maybe_aclose(provider: Any) -> None:
    close = getattr(provider, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def build_runtime(config: MikeConfig):
    ensure_root(config)
    bus = MessageBus()
    store = ChatStore(config)
    task_store = TaskStore(config.data_dir_path / "tasks")
    task_manager = TaskManager(task_store)
    research = ResearchManager(config, bus, task_store, task_manager)
    provider = make_provider(config)
    loop = AgentLoop(bus=bus, provider=provider, config=config, store=store, research=research)
    telegram = TelegramBot(config, bus, store)
    server = OpencodeServer(config)
    return bus, loop, telegram, server, provider


@app.command()
def onboard(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    path = Path(config_path).expanduser().resolve() if config_path else default_config_path()
    if path.exists():
        config = load_config(path)
        save_config(config, path)
        console.print(f"[green]OK[/green] Refreshed config at {path}")
    else:
        config = MikeConfig()
        save_config(config, path)
        console.print(f"[green]OK[/green] Created config at {path}")
    ensure_root(config)
    console.print("\nNext steps:")
    console.print(f"  1. Add your Telegram token to [cyan]{path}[/cyan] as `telegram_token`")
    console.print(f"  2. Add your OpenCode Go API key to [cyan]{path}[/cyan] as `opencode_api_key`")
    console.print("  3. Start the gateway with [cyan]mike gateway[/cyan]")


@app.command()
def gateway(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Override Mike data directory"),
):
    path = Path(config_path).expanduser().resolve() if config_path else default_config_path()
    config = load_config(path)
    if data_dir:
        config.data_dir = data_dir
    if not path.exists():
        save_config(config, path)
        console.print(f"[green]OK[/green] Created config at {path}")
    console.print("Starting Mike gateway...")

    async def run() -> None:
        bus, loop, telegram, server, provider = build_runtime(config)
        try:
            await server.ensure_running()
            console.print(f"[green]OK[/green] OpenCode Serve: {config.opencode_server_url}")
            if config.telegram_enabled:
                console.print("[green]OK[/green] Telegram enabled")
            else:
                console.print(
                    "[yellow]Warning[/yellow] Telegram token missing; bot will not receive messages"
                )
            tasks = [asyncio.create_task(loop.run())]
            if config.telegram_enabled:
                tasks.append(asyncio.create_task(telegram.start()))
                tasks.append(asyncio.create_task(telegram.bridge_outbound()))
            await asyncio.gather(*tasks)
        finally:
            loop.stop()
            await telegram.stop()
            await server.stop()
            await _maybe_aclose(provider)

    asyncio.run(run())


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    path = Path(config_path).expanduser().resolve() if config_path else default_config_path()
    config = load_config(path)
    if not path.exists():
        save_config(config, path)
        console.print(f"[green]OK[/green] Created config at {path}")

    async def run_once() -> None:
        _bus, loop, _telegram, server, provider = build_runtime(config)
        try:
            await server.ensure_running()
            response = await loop.process_direct(message or "Hello", session_key=session_id)
            console.print(response)
        finally:
            loop.stop()
            await server.stop()
            await _maybe_aclose(provider)

    asyncio.run(run_once())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
