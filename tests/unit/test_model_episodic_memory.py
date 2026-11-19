"""Real SQLite/tool/Sleep ownership checks for Normal and CRACK model memory."""
from pathlib import Path

import pytest

from core.tools import AgentToolRegistry
from memory.db import EpisodicMemoryDB
from memory.model_db import model_episodic_db_path


NORMAL = {"repo_id": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", "name": "Bonsai 2"}
CRACK = {"repo_id": "dealignai/Bonsai-2-27B-CRACK-Ternary-JANG", "name": "Bonsai CRACK"}


def test_identity_survives_app_move_and_display_rename_but_changes_with_artifact_revision(tmp_path):
    base = tmp_path / "memory.db"
    original = model_episodic_db_path(base, {**NORMAL, "model_path": "/Applications/Old.app/models/normal"})
    moved = model_episodic_db_path(base, {**NORMAL, "name": "Renamed", "short_name": "New label",
                                       "model_path": "/Users/me/Applications/New.app/models/normal"})
    assert original == moved
    assert original != model_episodic_db_path(base, CRACK)
    assert original != model_episodic_db_path(base, {**NORMAL, "source_revision": "new-model-commit"})
    assert len({model_episodic_db_path(base, NORMAL, implementation=mode)
                for mode in ("legacy", "managed", "hybrid")}) == 3
    assert model_episodic_db_path(base, {**NORMAL, "filename": "a.gguf"}) != model_episodic_db_path(
        base, {**NORMAL, "filename": "b.gguf"})
    # Built-in GGUF presets use gguf_file, before a local artifact is resolved.
    selected = model_episodic_db_path(base, {**NORMAL, "gguf_file": "target-a.gguf"})
    other = model_episodic_db_path(base, {**NORMAL, "gguf_file": "target-b.gguf"})
    loaded = model_episodic_db_path(base, {**NORMAL, "gguf_file": "target-a.gguf",
                                         "model_path": "/different/install/target-a.gguf"})
    assert selected != other and selected == loaded


@pytest.mark.parametrize("mode", ["legacy", "managed", "hybrid"])
def test_chat_tools_export_and_sleep_cannot_read_other_model_or_old_shared_rows(tmp_path, mode):
    base = tmp_path / "memory.db"
    old = EpisodicMemoryDB(base)
    old.log_interaction("Old mixed-owner chat", "preserved", verified_reward=1)
    original = base.read_bytes()
    normal = EpisodicMemoryDB(model_episodic_db_path(base, NORMAL, implementation=mode))
    crack = EpisodicMemoryDB(model_episodic_db_path(base, CRACK, implementation=mode))
    normal.log_interaction("Normal-only secret", "normal answer", verified_reward=1)
    crack.log_interaction("CRACK-only secret", "crack answer", verified_reward=1)

    for db, own, other in ((normal, "Normal-only", "CRACK-only"), (crack, "CRACK-only", "Normal-only")):
        tools = AgentToolRegistry(db.db_path, workspace_dir=str(tmp_path))
        ok, output = tools.execute_tool("read_chat_history", {})
        assert ok and own in output and other not in output and "Old mixed-owner" not in output
        filename = own + ".md"
        assert tools.execute_tool("export_chat_history", {"filename": filename})[0]
        exported = (tmp_path / filename).read_text()
        assert own in exported and other not in exported and "Old mixed-owner" not in exported
        with db.sleep_queue_lock():
            token, rows = db.prepare_verified_sleep(own, 100)
        assert len(rows) == 1 and own in rows[0]["prompt"]
        db.release_verified_sleep(token)
    assert base.read_bytes() == original
    assert old.get_unconsolidated_memories()[0]["prompt"] == "Old mixed-owner chat"
