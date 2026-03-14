"""Model configuration utilities for handling API differences between models."""

from __future__ import annotations

from typing import Any

AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "kimi-k2.5": {
        "vision": True,
        "description": "Moonshot Kimi K2.5 — best overall performance with vision support",
        "api_type": "openai-compatible",
        "endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "reasoning_param": "reasoning_effort",
        "reasoning_value": "high",
    },
    "glm-5": {
        "vision": False,
        "description": "Zhipu GLM-5 — fast text-only model",
        "api_type": "openai-compatible",
        "endpoint": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "reasoning_param": "reasoning_effort",
        "reasoning_value": "high",
    },
    "minimax-m2.5": {
        "vision": False,
        "description": "MiniMax M2.5 — most cost effective (text-only)",
        "api_type": "anthropic-compatible",
        "endpoint": "/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
        "reasoning_param": "thinking",
        "reasoning_value": {"type": "enabled"},
    },
}

DEFAULT_MODEL = "kimi-k2.5"


def get_model_config(model_id: str) -> dict[str, Any] | None:
    """Get full model configuration.
    
    Returns the model configuration dict or None if model not found.
    """
    return AVAILABLE_MODELS.get(model_id)


def get_model_reasoning_config(model_id: str) -> dict[str, Any] | None:
    """Generate reasoning configuration for a given model.

    Returns the appropriate reasoning configuration based on the model's API type.
    - OpenAI-compatible models: {"type": "reasoning_effort", "value": "high"}
    - Anthropic-compatible models: {"type": "thinking", "value": {"type": "enabled"}}

    Returns None if model is not found or reasoning should not be enabled.
    """
    model_config = get_model_config(model_id)
    if model_config is None:
        return None

    api_type = model_config.get("api_type", "openai-compatible")
    reasoning_param = model_config.get("reasoning_param", "reasoning_effort")
    reasoning_value = model_config.get("reasoning_value")

    if reasoning_value is None:
        if api_type == "anthropic-compatible":
            reasoning_value = {"type": "enabled"}
        else:
            reasoning_value = "high"

    return {
        "type": reasoning_param,
        "value": reasoning_value,
    }