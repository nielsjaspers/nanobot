from mike.chat.models import clamp_max_tokens, get_model
from mike.chat.reasoning import build_reasoning_kwargs
from mike.tasks.research import build_opencode_reasoning_config


def test_kimi_uses_openai_transport():
    cfg = get_model("kimi-k2.5")
    assert cfg is not None
    assert cfg["api_type"] == "openai-compatible"
    assert cfg["endpoint"] == "/chat/completions"
    assert cfg["auth_header"] == "Authorization"
    assert cfg["reasoning_param"] == "reasoning_effort"
    reasoning = build_reasoning_kwargs("kimi-k2.5")
    assert reasoning["reasoning_effort"] == "high"
    assert reasoning["thinking"] is None


def test_glm_uses_openai_transport():
    cfg = get_model("glm-5")
    assert cfg is not None
    assert cfg["api_type"] == "openai-compatible"
    assert cfg["endpoint"] == "/chat/completions"
    assert cfg["auth_header"] == "Authorization"
    assert cfg["reasoning_param"] == "reasoning_effort"
    reasoning = build_reasoning_kwargs("glm-5")
    assert reasoning["reasoning_effort"] == "high"
    assert reasoning["thinking"] is None


def test_minimax_uses_anthropic_transport_and_budget():
    cfg = get_model("minimax-m2.5")
    assert cfg is not None
    assert cfg["api_type"] == "anthropic-compatible"
    assert cfg["endpoint"] == "/messages"
    assert cfg["auth_header"] == "x-api-key"
    assert cfg["reasoning_param"] == "thinking"
    reasoning = build_reasoning_kwargs("minimax-m2.5")
    assert reasoning["reasoning_effort"] is None
    assert reasoning["thinking"] == {"type": "enabled"}


def test_opencode_reasoning_config_mapping():
    assert build_opencode_reasoning_config("kimi-k2.5") == {
        "type": "reasoning_effort",
        "value": "high",
    }
    assert build_opencode_reasoning_config("glm-5") == {
        "type": "reasoning_effort",
        "value": "high",
    }
    assert build_opencode_reasoning_config("minimax-m2.5") == {
        "type": "thinking",
    }


def test_model_max_tokens_are_clamped_safely():
    assert clamp_max_tokens("minimax-m2.5", 250000) == 196000
    assert clamp_max_tokens("kimi-k2.5", 300000) == 260000
    assert clamp_max_tokens("glm-5", 250000) == 202000


def test_supported_model_ids_exist_for_alias_targets():
    assert get_model("minimax-m2.5") is not None
    assert get_model("kimi-k2.5") is not None
    assert get_model("glm-5") is not None
