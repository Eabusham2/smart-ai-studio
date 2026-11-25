"""Updater interaction and worker handoff without a display or model runtime."""
import os
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.app_update_ui import AppUpdateUI


class UpdateUITests(unittest.TestCase):
    def setUp(self):
        self.owner = threading.get_ident()
        self.app = SimpleNamespace(
            root=SimpleNamespace(after=Mock(return_value="timer-1"), after_idle=Mock(return_value="ack-1"),
                                 after_cancel=Mock(), bind=Mock()),
            settings=SimpleNamespace(backend="mock"),
            _event_queue=queue.Queue(), _ui_thread_ident=self.owner,
            btn_app_update=Mock(), _on_close_app=Mock(),
            is_generating=False, prompt_queue=[],
            engine=SimpleNamespace(speed_tuning_active=False), multimodal=SimpleNamespace(busy=False),
        )
        self.release = SimpleNamespace(tag="latest", commit="a" * 40,
                                       html_url="https://github.com/Eabusham2/smart-ai-studio/releases/tag/latest",
                                       total_bytes=128)
        self.updater = SimpleNamespace(current_label="latest (bbbbbbbbbbbb)", check=Mock(), prepare=Mock(),
                                       install=Mock(), discard=Mock())
        self.updater.check.return_value = SimpleNamespace(
            status="available", message="A newer release is available.", release=self.release)
        self.ui = AppUpdateUI(self.app, updater_factory=lambda: self.updater)

    def finish_worker(self):
        self.ui._worker.join(timeout=2)
        self.assertFalse(self.ui._worker.is_alive())
        while not self.app._event_queue.empty():
            kind, callback = self.app._event_queue.get_nowait()
            self.assertEqual(kind, "ui_callback")
            callback()

    def assert_worker(self):
        self.assertNotEqual(threading.get_ident(), self.owner)

    def finish_cleanup(self):
        for worker in self.ui._cleanup_workers:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def make_available(self):
        self.ui.check()
        self.finish_worker()

    def test_check_is_off_thread_and_only_owner_updates_button(self):
        def check():
            self.assert_worker()
            return SimpleNamespace(status="available", message="Ready", release=self.release)
        self.updater.check.side_effect = check
        self.ui.check()
        self.ui._worker.join(timeout=2)
        self.assertIsNone(self.ui._release)
        self.assertEqual(self.app.btn_app_update.configure.call_args.kwargs["text"], "↑ Checking…")
        self.finish_worker()
        self.assertIs(self.ui._release, self.release)
        self.assertEqual(self.app.btn_app_update.configure.call_args.kwargs["text"], "↑ Update available")
        self.app.root.after.assert_not_called()

    def test_repeated_checks_do_not_start_another_worker(self):
        gate = threading.Event()
        self.updater.check.side_effect = lambda: (
            gate.wait(2) and SimpleNamespace(status="current", message="Already current", release=None))
        self.ui.check()
        worker = self.ui._worker
        self.ui.check()
        self.assertIs(self.ui._worker, worker)
        gate.set()
        self.finish_worker()
        self.updater.check.assert_called_once()

    def test_current_ahead_and_unsupported_results_do_not_offer_install(self):
        for status in ("current", "ahead", "unsupported"):
            with self.subTest(status=status):
                self.updater.check.return_value = SimpleNamespace(status=status, message=status, release=None)
                self.ui.check()
                self.finish_worker()
                self.ui.update_and_restart()
                self.assertIsNone(self.ui._release)
        self.updater.prepare.assert_not_called()
        self.app._on_close_app.assert_not_called()

    def test_offline_failure_is_quiet_and_can_be_retried(self):
        self.updater.check.side_effect = OSError("network unavailable")
        self.ui.check()
        self.finish_worker()
        self.assertFalse(self.ui._busy)
        self.assertIn("network unavailable", self.ui._message)
        self.assertEqual(self.ui._current_label, self.updater.current_label)
        self.assertIsNone(self.ui._dialog)
        self.updater.check.side_effect = None
        self.make_available()
        self.assertIs(self.ui._release, self.release)

    def test_prepare_install_and_restart_happen_in_order_on_the_right_threads(self):
        self.make_available()
        prepared = object()
        helper = Mock()
        calls = []
        def prepare(release, progress):
            self.assert_worker()
            self.assertIs(release, self.release)
            calls.append("prepare")
            progress("Verifying downloaded files", 128, 128)
            return prepared
        def install(value):
            self.assert_worker()
            self.assertIs(value, prepared)
            calls.append("install")
            return helper
        def close():
            self.assertEqual(threading.get_ident(), self.owner)
            calls.append("close")
        self.updater.prepare.side_effect = prepare
        self.updater.install.side_effect = install
        self.app._on_close_app.side_effect = close
        self.ui.update_and_restart()
        self.finish_worker()
        self.finish_worker()
        self.assertEqual(calls, ["prepare", "install", "close"])
        helper.terminate.assert_not_called()
        self.assertEqual(self.ui._progress_value, 100)

    def test_active_work_prevents_preparation_and_restart(self):
        self.make_available()
        for field in ("is_generating", "_model_load_in_progress", "_speed_ui_running",
                      "_sleep_consolidating", "_model_unload_pending"):
            with self.subTest(field=field):
                setattr(self.app, field, True)
                self.ui.update_and_restart()
                setattr(self.app, field, False)
                self.updater.prepare.assert_not_called()
        self.app.prompt_queue = [("pending",)]
        self.ui.update_and_restart()
        self.updater.prepare.assert_not_called()
        self.app._on_close_app.assert_not_called()

    def test_new_work_during_download_keeps_prepared_update_for_later(self):
        self.make_available()
        prepared = object()
        def prepare(*_):
            self.app.is_generating = True
            return prepared
        self.updater.prepare.side_effect = prepare
        self.ui.update_and_restart()
        self.finish_worker()
        self.assertIs(self.ui._prepared, prepared)
        self.assertFalse(self.ui._busy)
        self.updater.install.assert_not_called()
        self.app._on_close_app.assert_not_called()
        self.app.is_generating = False
        self.ui.update_and_restart()
        self.finish_worker()
        self.updater.prepare.assert_called_once()
        self.updater.install.assert_called_once_with(prepared)
        self.app._on_close_app.assert_called_once()

    def test_prompts_queued_for_an_inactive_model_prevent_restart(self):
        self.make_available()
        self.app._model_prompt_queues = {"active": [], "other-model": [("pending prompt",)]}
        self.ui.update_and_restart()
        self.updater.prepare.assert_not_called()
        self.app._on_close_app.assert_not_called()

    def test_same_commit_with_changed_release_files_discards_prepared_update(self):
        self.make_available()
        prepared = self.ui._prepared = object()
        changed = SimpleNamespace(**vars(self.release))
        changed.total_bytes += 1
        self.updater.check.return_value = SimpleNamespace(status="available", message="Updated assets", release=changed)
        self.updater.discard.side_effect = lambda _: self.assert_worker()
        self.ui.check()
        self.finish_worker()
        self.finish_cleanup()
        self.assertIsNone(self.ui._prepared)
        self.assertIs(self.ui._release, changed)
        self.updater.discard.assert_called_once_with(prepared)

    def test_unchanged_release_retains_prepared_update(self):
        self.make_available()
        prepared = self.ui._prepared = object()
        self.ui.check()
        self.finish_worker()
        self.assertIs(self.ui._prepared, prepared)
        self.updater.discard.assert_not_called()

    def test_close_discards_idle_prepared_update_off_thread(self):
        self.make_available()
        prepared = self.ui._prepared = object()
        self.updater.discard.side_effect = lambda _: self.assert_worker()
        self.ui._on_destroy(SimpleNamespace(widget=self.app.root))
        self.finish_cleanup()
        self.updater.discard.assert_called_once_with(prepared)

    def test_close_does_not_discard_staging_owned_by_installer(self):
        self.make_available()
        self.ui._prepared = object()
        self.ui._helper_owned = True
        self.ui._on_destroy(SimpleNamespace(widget=self.app.root))
        self.updater.discard.assert_not_called()

    def test_work_started_during_handoff_stops_waiting_helper(self):
        self.make_available()
        helper = Mock()
        def install(_):
            self.app.multimodal.busy = True
            return helper
        self.updater.install.side_effect = install
        self.ui.update_and_restart()
        for _ in range(3):
            self.finish_worker()
        helper.terminate.assert_called_once()
        helper.wait.assert_called_once_with(timeout=10)
        self.assertFalse(self.ui._busy)
        self.assertIn("restart was deferred", self.ui._message)
        self.app._on_close_app.assert_not_called()

    def test_failure_during_download_does_not_close_or_install(self):
        self.make_available()
        self.updater.prepare.side_effect = ValueError("Checksum mismatch")
        self.ui.update_and_restart()
        self.finish_worker()
        self.updater.install.assert_not_called()
        self.app._on_close_app.assert_not_called()
        self.assertIn("Checksum mismatch", self.ui._message)
        self.assertFalse(self.ui._busy)

    def test_destroyed_window_discards_worker_result(self):
        self.ui.check()
        self.ui._worker.join(timeout=2)
        self.app._ui_poll_closed = True
        self.finish_worker()
        self.assertIsNone(self.ui._release)

    def test_mock_or_offline_mode_does_not_schedule_automatic_network_check(self):
        with patch.object(self.ui, "_acknowledge_startup"):
            self.ui.start()
            self.app.root.after_idle.call_args.args[0]()
            self.ui._startup_worker.join(timeout=2)
        self.app.root.after.assert_not_called()
        self.app.settings.backend = "mlx"
        with patch.dict(os.environ, {"SMARTAI_TEST_OFFLINE": "1"}), patch.object(self.ui, "_acknowledge_startup"):
            self.ui.start()
            self.app.root.after_idle.call_args.args[0]()
            self.ui._startup_worker.join(timeout=2)
        self.app.root.after.assert_not_called()

    def test_normal_startup_schedules_quiet_check_and_destroy_cancels_it(self):
        self.app.settings.backend = "mlx"
        with patch.dict(os.environ, {"SMARTAI_TEST_OFFLINE": "0"}), patch.object(self.ui, "_acknowledge_startup"):
            self.ui.start()
            self.app.root.after_idle.call_args.args[0]()
            self.ui._startup_worker.join(timeout=2)
        self.app.root.after.assert_called_once_with(1500, self.ui._automatic_check)
        self.ui._on_destroy(SimpleNamespace(widget=self.app.root))
        self.app.root.after_cancel.assert_called_once_with("timer-1")
        self.ui.check()
        self.updater.check.assert_not_called()

    def test_acknowledgement_waits_for_tk_idle_and_runs_off_thread(self):
        with patch.object(self.ui, "_acknowledge_startup", side_effect=self.assert_worker) as acknowledge:
            self.ui.start()
            acknowledge.assert_not_called()
            self.assertIsNone(self.ui._startup_worker)
            self.app.root.after_idle.call_args.args[0]()
            self.ui._startup_worker.join(timeout=2)
            acknowledge.assert_called_once()

    def test_destroy_before_idle_does_not_acknowledge_startup(self):
        with patch.object(self.ui, "_acknowledge_startup") as acknowledge:
            self.ui.start()
            self.ui._on_destroy(SimpleNamespace(widget=self.app.root))
            self.app.root.after_cancel.assert_called_once_with("ack-1")
            self.app.root.after_idle.call_args.args[0]()
            acknowledge.assert_not_called()

    def test_failed_helper_stop_keeps_handoff_exclusive_and_preserves_staging(self):
        self.make_available()
        helper = Mock()
        helper.terminate.side_effect = OSError("Cannot terminate helper")
        def install(_):
            self.app.is_generating = True
            return helper
        self.updater.install.side_effect = install
        with self.assertLogs("core.app_update_ui", level="ERROR"):
            self.ui.update_and_restart()
            for _ in range(3):
                self.finish_worker()
        self.assertTrue(self.ui._helper_owned)
        self.assertTrue(self.ui._busy)
        self.app._on_close_app.assert_not_called()
        self.ui._on_destroy(SimpleNamespace(widget=self.app.root))
        self.updater.discard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
