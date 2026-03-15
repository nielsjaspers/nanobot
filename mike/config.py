"""Minimal configuration for Mike."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class MikeConfig(BaseModel):
    telegram_token: str = ""
    telegram_allow_from: list[str] = Field(default_factory=list)
    telegram_proxy: str | None = None
    telegram_reply_to_message: bool = False
    telegram_group_policy: str = "mention"

    opencode_api_key: str = ""
    opencode_api_base: str = "https://opencode.ai/zen/go/v1"
    opencode_server_url: str = "http://127.0.0.1:4096"
    opencode_server_password: str = ""
    opencode_server_bin: str = "opencode"
    opencode_server_autostart: bool = True
    opencode_server_host: str = "127.0.0.1"
    opencode_server_port: int = 4096
    opencode_model_provider_id: str = "opencode-go"
    opencode_agent: str | None = None

    default_model: str = "kimi-k2.5"
    data_dir: str = "~/.mike"
    project_root: str = "."
    send_progress: bool = True
    send_tool_hints: bool = False
    max_tool_iterations: int = 24
    max_tokens: int = 8192
    temperature: float = 1.0
    command_timeout: int = 120
    restrict_shell_to_project: bool = False

    skills_dir: str = ".opencode/skills"
    deep_research_skill_path: str = "mike/resources/deep-research/SKILL.md"

    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def project_root_path(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token)


def default_config_path() -> Path:
    override = os.environ.get("MIKE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".mike" / "config.json"


def default_config() -> MikeConfig:
    return MikeConfig()


def load_config(path: Path | None = None) -> MikeConfig:
    path = path or default_config_path()
    if not path.exists():
        return default_config()
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("minimax_budget_tokens", None)
    return MikeConfig.model_validate(data)


def save_config(config: MikeConfig, path: Path | None = None) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, ensure_ascii=False, indent=2)
    return path
