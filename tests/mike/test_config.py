from mike.config import MikeConfig


def test_config_defaults_are_minimal_and_easy():
    cfg = MikeConfig()
    assert cfg.telegram_token == ""
    assert cfg.opencode_api_base == "https://opencode.ai/zen/go/v1"
    assert cfg.opencode_server_url == "http://127.0.0.1:4096"
    assert cfg.default_model == "kimi-k2.5"
    assert cfg.opencode_server_autostart is True
