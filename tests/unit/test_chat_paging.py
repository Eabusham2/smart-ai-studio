from core.chat_paging import export_transcript, transcript_page
from core.context_memory import ChatArchive
from tools.online_learning import read_request
import pytest


def test_paging_reads_only_the_selected_range_and_all_pages_are_reachable(tmp_path, monkeypatch):
    archive = ChatArchive(tmp_path / "transcript.sqlite3", "chat")
    for index in range(140):
        archive.append({"role": "user", "content": str(index) + ":" + "z" * 3000})
    decoded = []
    original = ChatArchive._decode

    def decode(row):
        decoded.append(row[0])
        return original(row)

    monkeypatch.setattr(ChatArchive, "_decode", staticmethod(decode))
    latest = transcript_page(archive, max_turns=20, max_chars=8000, message_chars=4000)
    assert latest.end == 140 and latest.start == 138
    assert len(decoded) == 20
    previous = transcript_page(archive, end=latest.start, max_turns=20, max_chars=8000)
    following = transcript_page(archive, start=previous.end, max_turns=20, max_chars=8000)
    assert previous.end == following.start and following.end == latest.end
    assert sum(len(turn["content"]) for turn in following.turns) <= 8000


def test_long_message_preview_does_not_modify_archived_or_exported_text(tmp_path):
    archive = ChatArchive(tmp_path / "transcript.sqlite3", "chat")
    content = "important " * 5000 + "EXACT_FINAL_FACT"
    archive.append({"role": "assistant", "content": content})
    page = transcript_page(archive)
    assert page.previewed and "Long message preview" in page.turns[0]["content"]
    assert archive[0]["content"] == content
    destination = tmp_path / "conversation.md"
    export_transcript(archive, destination)
    assert content in destination.read_text()


def test_cli_request_is_bounded_json_data(tmp_path, monkeypatch):
    from tools import online_learning
    request = tmp_path / "request.json"
    request.write_text('{"method":"ingest","source":"source.txt"}')
    assert read_request(request)["source"] == "source.txt"
    request.write_text('["method", "ingest"]')
    with pytest.raises(ValueError, match="object"):
        read_request(request)
    request.write_text("x" * 100)
    monkeypatch.setattr(online_learning, "MAX_REQUEST_BYTES", 40)
    with pytest.raises(ValueError, match="large documents"):
        read_request(request)


def test_gui_scope_preference_survives_restart_and_never_saves_request(tmp_path):
    from core.memory_preferences import load_memory_preference, save_memory_preference
    from core.memory_scopes import LearningScope
    path = tmp_path / "chat.sqlite3"
    assert load_memory_preference(path) == LearningScope()
    selected = LearningScope("project", "renderer")
    save_memory_preference(path, selected)
    assert load_memory_preference(path) == selected
    with pytest.raises(ValueError, match="durable"):
        save_memory_preference(path, LearningScope.request())
    assert load_memory_preference(path) == selected
