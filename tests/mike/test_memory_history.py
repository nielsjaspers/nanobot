import json
from pathlib import Path

from mike.config import MikeConfig
from mike.storage.chats import ChatSession, ChatStore
from mike.memory.search import search_index, search_memory_sections


def test_shared_files_exist_and_sessions_are_separate(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"))
    store = ChatStore(cfg)
    assert store.soul_path().exists()
    assert store.user_path().exists()
    assert store.memory_path().exists()
    assert store.history_index_path().exists()
    assert store.session_root("cli:direct").name == "cli_direct"
    assert store.session_root("telegram:123").name == "telegram_123"


def test_session_reset_clears_active_transcript(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"))
    store = ChatStore(cfg)
    session = store.get("cli:direct")
    session.messages.append(
        {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00"}
    )
    session.current_model = "glm-5"
    store.save(session)
    reset = store.reset("cli:direct")
    assert reset.messages == []
    assert reset.current_model == "glm-5"
    payload = json.loads(store.state_path("cli:direct").read_text(encoding="utf-8"))
    assert payload["messages"] == []


def test_session_reset_can_drop_model_when_requested(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"))
    store = ChatStore(cfg)
    session = store.get("cli:direct")
    session.current_model = "minimax-m2.5"
    store.save(session)
    reset = store.reset("cli:direct", preserve_model=False)
    assert reset.current_model is None


def test_search_helpers_work_for_memory_and_history(tmp_path: Path):
    cfg = MikeConfig(data_dir=str(tmp_path / "mike-data"))
    store = ChatStore(cfg)
    store.memory_path().write_text(
        "# Long-term Memory\n\n## Preferences\n\nUser likes concise summaries.\n\n## Project Context\n\nMike bot rewrite is ongoing.\n",
        encoding="utf-8",
    )
    store.history_index_path().write_text(
        json.dumps(
            [
                {
                    "id": "abc123",
                    "title": "Mike rewrite planning",
                    "summary": "Discussed shared memory, archive JSON, and /new behavior.",
                    "archived_at": "2026-03-15T12:00:00",
                    "metadata": {"channel": "telegram", "chat_id": "1"},
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    matches = search_index(store.history_index_path(), "archive shared memory", limit=5)
    assert matches and matches[0].archive_id == "abc123"
    memory = search_memory_sections(store.memory_path(), "concise summaries", max_chars=500)
    assert "User likes concise summaries." in memory
