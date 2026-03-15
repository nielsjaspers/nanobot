from mike.memory.archive import ArchiveManager
from mike.storage.chats import ChatSession


def test_archive_json_parser_handles_fenced_json():
    payload = '```json\n{"title":"A","summary":"B","memory_update":"C"}\n```'
    parsed = ArchiveManager._parse_archive_json(payload)
    assert parsed == {"title": "A", "summary": "B", "memory_update": "C"}


def test_archive_fallback_summary_preserves_memory():
    session = ChatSession(
        key="telegram:1",
        messages=[
            {
                "role": "user",
                "content": "We talked about project plans",
                "timestamp": "2026-03-15T12:00:00",
            },
            {
                "role": "assistant",
                "content": "I suggested a smaller architecture",
                "timestamp": "2026-03-15T12:00:10",
            },
        ],
    )
    title, summary, memory_update = ArchiveManager._fallback_summary(
        session, "# Long-term Memory\n"
    )
    assert title.startswith("We talked about project plans")
    assert "Archived conversation" in summary
    assert memory_update == "# Long-term Memory\n"
