"""The default desktop Sleep control still runs the original standalone EWC daemon."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import app_gui
from config.settings import Settings
from memory.db import EpisodicMemoryDB


def test_original_sleep_action_uses_daemon_and_stale_dialog_cannot_start_it(tmp_path, monkeypatch):
    settings = Settings(database_path=str(tmp_path / "episodes.sqlite3"), use_mock=True)
    assert settings.learning_implementation == "legacy"
    db = EpisodicMemoryDB(db_path=settings.database_path)
    episode_ids = [db.log_interaction(f"prompt-{i}", "answer", verified_reward=1, surprise_score=0.7)
                   for i in range(51)]
    db.log_interaction("unverified", "answer", verified_reward=0, surprise_score=1.0)

    root = Mock()
    root.after.side_effect = lambda delay, callback: callback()
    log, button = Mock(), Mock()
    # Use the public desktop command, replacing only its display, executor and trainer.
    for widget in ("Toplevel", "Label", "Frame"):
        monkeypatch.setattr(app_gui.tk, widget, MagicMock())
    monkeypatch.setattr(app_gui.tk, "Text", Mock(return_value=log))
    monkeypatch.setattr(app_gui.tk, "Button", Mock(return_value=button))
    monkeypatch.setattr(app_gui.threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    daemon = Mock()
    monkeypatch.setattr(app_gui, "SleepConsolidationDaemon", daemon)
    badges = [Mock(), Mock(), Mock()]
    app = SimpleNamespace(root=root, settings=settings, C=defaultdict(str), models_config={"tab": {}},
                          active_tab_id="tab", engine=SimpleNamespace(), _make_badge=Mock(side_effect=badges),
                          _update_telemetry=Mock())

    checkpoint = str(tmp_path / "learned.safetensors")
    def finish_cycle():
        assert app._sleep_consolidating
        db.mark_consolidated(episode_ids[:50])
        return {"status": "success", "memories_consolidated": 50, "avg_task_loss": 0.32, "avg_ewc_loss": 0.125,
                "adapter_saved_to": checkpoint, "execution_time_seconds": 3.5}

    daemon.return_value.run_consolidation_cycle.side_effect = finish_cycle
    app_gui.SmartAIChatbotApp._on_open_sleep_consolidation_panel(app)
    badge_label = app._make_badge.call_args_list[0].args[1]
    assert "Pending verified episodes: 51" in badge_label
    assert "Pending Traces: 51" in log.insert.call_args.args[1]

    trigger = button.configure.call_args.kwargs["command"]
    trigger()
    daemon.assert_called_once_with(settings=settings, use_mock=True)
    daemon.return_value.run_consolidation_cycle.assert_called_once_with()
    assert db.get_stats()["unconsolidated_verified"] == 1
    assert "Pending verified episodes: 1" in badges[0].configure.call_args.kwargs["text"]
    assert app._sleep_consolidating is False
    app._update_telemetry.assert_called_once_with()
    report = "".join(call.args[1] for call in log.insert.call_args_list)
    for result in ("Memories Consolidated: 50 (3.50s)", "0.3200", "0.1250", checkpoint):
        assert result in report
    assert button.configure.call_args.kwargs["state"] == "normal"

    settings.learning_implementation = "managed"
    trigger()
    daemon.return_value.run_consolidation_cycle.assert_called_once_with()
    assert "reopen Sleep" in log.insert.call_args.args[1]
