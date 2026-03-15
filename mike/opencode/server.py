"""Local OpenCode Serve process management for Mike."""

from __future__ import annotations

import asyncio
import contextlib
import os

from loguru import logger

from mike.config import MikeConfig
from mike.opencode.client import OpencodeClient


class OpencodeServer:
    def __init__(self, config: MikeConfig):
        self.config = config
        self.proc: asyncio.subprocess.Process | None = None

    async def ensure_running(self) -> None:
        if await self.is_healthy():
            logger.info(
                "Mike attached to existing OpenCode server at {}", self.config.opencode_server_url
            )
            return
        if not self.config.opencode_server_autostart:
            logger.warning(
                "OpenCode server at {} is unavailable and autostart is disabled",
                self.config.opencode_server_url,
            )
            return
        if self.proc and self.proc.returncode is None:
            return
        env = os.environ.copy()
        if self.config.opencode_server_password:
            env["OPENCODE_SERVER_PASSWORD"] = self.config.opencode_server_password
        self.proc = await asyncio.create_subprocess_exec(
            self.config.opencode_server_bin,
            "serve",
            "--hostname",
            self.config.opencode_server_host,
            "--port",
            str(self.config.opencode_server_port),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info(
            "Mike started a new OpenCode server at {}",
            self.config.opencode_server_url,
        )
        for _ in range(30):
            if await self.is_healthy():
                return
            await asyncio.sleep(1)
        raise RuntimeError("OpenCode server did not become healthy in time")

    async def is_healthy(self) -> bool:
        try:
            client = OpencodeClient(
                base_url=self.config.opencode_server_url,
                password=self.config.opencode_server_password,
                directory=str(self.config.project_root_path),
                username="opencode",
                timeout=15,
            )
            try:
                await client.health()
                return True
            finally:
                await client.aclose()
        except Exception:
            return False

    async def stop(self) -> None:
        if not self.proc or self.proc.returncode is not None:
            return
        self.proc.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.proc.wait(), timeout=5)
