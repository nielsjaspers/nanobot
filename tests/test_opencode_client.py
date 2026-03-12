from nanobot.opencode_client import OpencodeServeClient


def test_extract_text_reads_flat_text_parts() -> None:
    payload = {
        "info": {"id": "m1"},
        "parts": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ],
    }

    assert OpencodeServeClient.extract_text(payload) == "Hello\nWorld"


def test_extract_text_reads_nested_data_parts() -> None:
    payload = {
        "data": {"parts": [{"type": "message", "parts": [{"type": "text", "text": "Nested text"}]}]}
    }

    assert OpencodeServeClient.extract_text(payload) == "Nested text"


def test_extract_text_falls_back_to_title() -> None:
    payload = {"info": {"title": "Result title"}, "parts": []}

    assert OpencodeServeClient.extract_text(payload) == "Result title"
