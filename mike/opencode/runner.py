"""Thin wrapper for `opencode run --attach`."""

from __future__ import annotations

import asyncio


class OpencodeRunner:
    def __init__(self, binary: str, attach_url: str):
        self.binary = binary
        self.attach_url = attach_url

    async def run(
        self, prompt: str, model: str | None = None, provider_id: str = "opencode-go"
    ) -> tuple[str, str, int | None]:
        cmd = [self.binary, "run", "--format", "json", "--attach", self.attach_url]
        if model:
            cmd.extend(["--model", f"{provider_id}/{model}"])
        cmd.append(prompt)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (
            out.decode("utf-8", errors="replace") if out else "",
            err.decode("utf-8", errors="replace") if err else "",
            proc.returncode,
        )
