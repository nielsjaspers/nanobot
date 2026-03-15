"""Provider wrapper for Mike."""

from __future__ import annotations

from mike.chat.models import DEFAULT_MODEL
from mike.config import MikeConfig
from mike.custom_provider import CustomProvider
from mike.llm import GenerationSettings, LLMProvider


def make_provider(config: MikeConfig) -> LLMProvider:
    provider = CustomProvider(
        api_key=config.opencode_api_key or "no-key",
        api_base=config.opencode_api_base,
        default_model=config.default_model or DEFAULT_MODEL,
    )
    provider.generation = GenerationSettings(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reasoning_effort=None,
    )
    return provider
