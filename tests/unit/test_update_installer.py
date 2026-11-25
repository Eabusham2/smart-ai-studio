"""Model-free update handoff tests, including real OS-helper transactions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from core import update_installer as installer


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="smartai-install-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.install = self.root / "Smart AI's $(touch INJECTED)"
        self.stage = self.root / "incoming" / "Smart AI"
        self.clean_env = mock.patch.dict(os.environ, {}, clear=False)
        self.clean_env.start()
        self.addCleanup(self.clean_env.stop)
        os.environ.pop("SMARTAI_BOOTSTRAP_PID", None)
        os.environ.pop("SMARTAI_DATA_DIR", None)

    def package(self, root, *, commit="a", fail=False, relative="."):
        code = root / relative
        (code / "core").mkdir(parents=True)
        shutil.copy2(installer.__file__, code / "core/update_installer.py")
        (code / "core/__init__.py").write_text("", encoding="utf-8")
        python = code / "runtime/python" / ("python.exe" if os.name == "nt" else "bin/python3")
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n', encoding="utf-8")
        python.chmod(0o755)
        bootstrap = (
            "from pathlib import Path\n"
            "import sys\n"
            "root=Path(__file__).resolve().parent\n"
            "sys.path.insert(0,str(root))\n"
            "from core.update_installer import acknowledge_startup\n"
            "(root/'launched.txt').write_text('" + commit + "')\n"
            + ("raise SystemExit(23)\n" if fail else "acknowledge_startup(root)\n")
        )
        (code / "bootstrap_runtime.py").write_text(bootstrap, encoding="utf-8")
        (code / "package.txt").write_text(commit, encoding="utf-8")
        managed = ["core", "runtime", "bootstrap_runtime.py", "package.txt", installer.BUILD_RECORD]
        record = {
            "schema": 1, "repository": installer.REPOSITORY, "commit": commit * 40,
            "target": "Linux-x86_64", "tag": "latest", "package_kind": "portable",
            "managed_paths": managed,
            "install_managed_paths": managed if relative == "." else [relative.split("/")[0]],
        }
        (code / installer.BUILD_RECORD).write_text(json.dumps(record), encoding="utf-8")
        return code

    def user_files(self, code):
        for name, contents in {
            "data/memory.db": b"personal memory",
            "preloaded_models/model/weights.bin": b"model weights",
            "my-work/draft.txt": b"my draft",
            ".env": b"PRIVATE_SETTING=kept",
        }.items():
            path = code / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

    def process(self, **kwargs):
        process = installer.schedule_install(self.install, self.stage, ".", gui_pid=0, **kwargs)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        return process

    @staticmethod
    def wait_status(process, expected, timeout=5):
        end = time.monotonic() + timeout
        path = process.smartai_update_directory / "result.json"
        while time.monotonic() < end:
            try:
                if json.loads(path.read_text(encoding="utf-8-sig"))["status"] == expected:
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        raise AssertionError(f"Helper did not reach {expected}: {path.read_text() if path.exists() else 'no status'}")

    @unittest.skipIf(os.name == "nt", "POSIX helper integration; Windows helper is separately checked")
    def test_real_install_preserves_user_data_and_model_inode(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        self.user_files(self.install)
        inode = (self.install / "preloaded_models/model/weights.bin").stat().st_ino
        # The release can include a fresh scaffold; existing user data wins.
        (self.stage / "data").mkdir()
        (self.stage / "data/fresh-scaffold").write_text("fresh")
        record_path = self.install / installer.BUILD_RECORD
        record = json.loads(record_path.read_text())
        record["managed_paths"].append("obsolete.py")
        record_path.write_text(json.dumps(record))
        (self.install / "obsolete.py").write_text("old packaged code")
        process = self.process()
        self.assertEqual(process.wait(timeout=10), 0, (process.smartai_update_directory / "helper.log").read_text())
        self.wait_status(process, "complete")
        self.assertEqual((self.install / "package.txt").read_text(), "b")
        self.assertEqual((self.install / "data/memory.db").read_bytes(), b"personal memory")
        self.assertEqual((self.install / "preloaded_models/model/weights.bin").stat().st_ino, inode)
        self.assertEqual((self.install / "my-work/draft.txt").read_bytes(), b"my draft")
        self.assertEqual((self.install / ".env").read_bytes(), b"PRIVATE_SETTING=kept")
        self.assertFalse((self.install / "obsolete.py").exists())
        self.assertFalse((self.install / installer.PENDING_RECORD).exists())
        self.assertFalse((self.root / "INJECTED").exists())
        plan = json.loads((process.smartai_update_directory / "plan.json").read_text())
        backup = Path(plan["backup"])
        self.assertEqual((backup / "package.txt").read_text(), "a")
        self.assertTrue((backup / "obsolete.py").is_file())
        self.assertFalse((backup / "preloaded_models").exists())

    @unittest.skipIf(os.name == "nt", "POSIX helper integration")
    def test_new_application_failure_rolls_back_data_and_relaunches_old(self):
        self.package(self.install)
        self.package(self.stage, commit="b", fail=True)
        self.user_files(self.install)
        process = self.process()
        self.assertEqual(process.wait(timeout=10), 1)
        self.wait_status(process, "failed")
        self.assertEqual((self.install / "package.txt").read_text(), "a")
        self.assertEqual((self.install / "data/memory.db").read_bytes(), b"personal memory")
        self.assertEqual((self.install / "preloaded_models/model/weights.bin").read_bytes(), b"model weights")
        end = time.monotonic() + 5
        while not (self.install / "launched.txt").exists() and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual((self.install / "launched.txt").read_text(), "a")
        self.assertTrue((process.smartai_update_directory / "failed-package").is_dir())

    @unittest.skipIf(os.name == "nt", "POSIX helper integration")
    def test_cancelling_waiting_helper_leaves_running_install_untouched(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        process = installer.schedule_install(self.install, self.stage, ".", gui_pid=os.getpid())
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.wait_status(process, "waiting")
        process.terminate()
        self.assertNotEqual(process.wait(timeout=5), 0)
        self.assertEqual((self.install / "package.txt").read_text(), "a")
        self.assertEqual((self.stage / "package.txt").read_text(), "b")
        plan = json.loads((process.smartai_update_directory / "plan.json").read_text())
        self.assertFalse(Path(plan["backup"]).exists())

    @unittest.skipIf(os.name == "nt", "POSIX helper integration")
    def test_disappearing_staged_package_restores_previous_install(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        blocker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: blocker.poll() is None and blocker.kill())
        process = installer.schedule_install(self.install, self.stage, ".", gui_pid=blocker.pid)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.wait_status(process, "waiting")
        self.stage.rename(self.stage.with_name("moved-away"))
        blocker.terminate()
        blocker.wait(timeout=5)
        self.assertEqual(process.wait(timeout=10), 1)
        self.wait_status(process, "failed")
        self.assertEqual((self.install / "package.txt").read_text(), "a")

    @unittest.skipIf(os.name == "nt", "POSIX helper integration")
    def test_initial_rename_failure_reopens_unchanged_application(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        blocker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: blocker.poll() is None and blocker.kill())
        commands = self.root / "commands"
        commands.mkdir()
        move = commands / "mv"
        move.write_text("#!/bin/sh\nif [ \"$1\" = " + shlex.quote(str(self.install))
                        + " ]; then exit 1; fi\nexec /bin/mv \"$@\"\n")
        move.chmod(0o755)
        with mock.patch.dict(os.environ, {"PATH": str(commands) + os.pathsep + os.environ.get("PATH", "")}):
            process = installer.schedule_install(self.install, self.stage, ".", gui_pid=blocker.pid)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.wait_status(process, "waiting")
        blocker.terminate()
        blocker.wait(timeout=5)
        self.assertEqual(process.wait(timeout=10), 1)
        self.wait_status(process, "failed")
        end = time.monotonic() + 5
        while not (self.install / "launched.txt").exists() and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual((self.install / "launched.txt").read_text(), "a")
        self.assertEqual((self.install / "package.txt").read_text(), "a")

    @unittest.skipIf(os.name == "nt", "POSIX helper integration")
    def test_real_linux_layout_keeps_install_and_code_user_folders(self):
        relative = "SmartAI.AppDir/usr/bin"
        old_code = self.package(self.install, relative=relative)
        self.package(self.stage, commit="b", relative=relative)
        self.user_files(old_code)
        (self.install / "user-notes.txt").write_text("keep notes")
        process = installer.schedule_install(self.install, self.stage, relative, gui_pid=0)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.assertEqual(process.wait(timeout=10), 0, (process.smartai_update_directory / "helper.log").read_text())
        self.assertEqual((old_code / "package.txt").read_text(), "b")
        self.assertEqual((old_code / "data/memory.db").read_bytes(), b"personal memory")
        self.assertEqual((self.install / "user-notes.txt").read_text(), "keep notes")

    def test_helper_start_failure_is_reported_before_app_shutdown(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        failed = mock.Mock()
        failed.poll.return_value = 1
        with mock.patch.object(installer.subprocess, "Popen", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "application is still open"):
                installer.schedule_install(self.install, self.stage, ".", gui_pid=os.getpid())
        self.assertEqual((self.install / "package.txt").read_text(), "a")
        self.assertFalse((self.stage / installer.PENDING_RECORD).exists())

    def test_nested_package_preserves_code_and_install_level_user_content(self):
        relative = "Contents/Resources/app"
        old_code = self.package(self.install, relative=relative)
        self.package(self.stage, commit="b", relative=relative)
        self.user_files(old_code)
        (self.install / "personal-notes.txt").write_text("keep")
        paths = installer.preservation_paths(self.install, self.stage, relative)
        self.assertIn("Contents/Resources/app/data", paths)
        self.assertIn("Contents/Resources/app/preloaded_models", paths)
        self.assertIn("personal-notes.txt", paths)
        self.assertNotIn("Contents", paths)

    def test_explicit_data_inside_managed_directory_is_preserved(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        custom = self.install / "core/user-database"
        custom.mkdir()
        os.environ["SMARTAI_DATA_DIR"] = str(custom)
        self.assertIn("core/user-database", installer.preservation_paths(self.install, self.stage, "."))

    def test_data_root_cannot_replace_new_application_code_or_runtime(self):
        relative = "Contents/Resources/app"
        code = self.package(self.install, relative=relative)
        self.package(self.stage, commit="b", relative=relative)
        for data in (self.install, self.install / "Contents", code, code / "core", code / "runtime"):
            with self.subTest(data=data):
                os.environ["SMARTAI_DATA_DIR"] = str(data)
                with self.assertRaisesRegex(RuntimeError, "data directory"):
                    installer.preservation_paths(self.install, self.stage, relative)

    def test_manifest_traversal_is_rejected_before_any_replacement(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        path = self.install / installer.BUILD_RECORD
        record = json.loads(path.read_text())
        record["managed_paths"].append("../escape")
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            self.process()
        self.assertEqual((self.install / "package.txt").read_text(), "a")

    def test_git_checkout_is_not_overwritten(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        (self.install / ".git").write_text("gitdir: elsewhere")
        with self.assertRaisesRegex(RuntimeError, "Git checkout"):
            self.process()

    def test_invalid_code_layout_and_overlapping_stage_are_rejected(self):
        self.package(self.install)
        self.package(self.stage, commit="b")
        with self.assertRaisesRegex(RuntimeError, "layout"):
            installer.schedule_install(self.install, self.stage, "../../outside", gui_pid=0)
        with self.assertRaisesRegex(RuntimeError, "destination"):
            installer.schedule_install(self.install, self.install, ".", gui_pid=0)

    def test_acknowledgement_rejects_unrelated_path_and_wrong_build(self):
        self.package(self.install)
        token = "1" * 32
        wrong = self.root / "unrelated" / ("ready-" + token)
        wrong.parent.mkdir()
        pending = self.install / installer.PENDING_RECORD
        record = {"ack_path": str(wrong), "token": token, "commit": "a" * 40}
        pending.write_text(json.dumps(record))
        self.assertFalse(installer.acknowledge_startup(self.install))
        self.assertFalse(wrong.exists())
        transaction = self.root / ".smartai-update-fixture"
        transaction.mkdir()
        right = transaction / ("ready-" + token)
        record.update(ack_path=str(right), commit="b" * 40)
        pending.write_text(json.dumps(record))
        self.assertFalse(installer.acknowledge_startup(self.install))
        self.assertFalse(right.exists())
        record["commit"] = "a" * 40
        pending.write_text(json.dumps(record))
        self.assertTrue(installer.acknowledge_startup(self.install))
        self.assertEqual(right.read_text().strip(), token)

    def test_windows_helper_loads_data_as_json_and_uses_os_process(self):
        # This is structural coverage, not a claim of native Windows execution.
        # It protects against interpreting paths as PowerShell expressions and
        # against an already-exited child preventing the rollback transaction.
        script = installer._POWERSHELL
        self.assertIn("Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json", script)
        self.assertNotIn("Invoke-Expression", script)
        self.assertIn("WaitForExit", script)
        self.assertIn("taskkill.exe", script)
        self.assertIn("if (-not $script:child.HasExited)", script)
        self.assertIn("$ErrorActionPreference = 'Continue'", script)
        self.assertIn("$ErrorActionPreference = $previousPreference", script)
        self.assertIn("Move-Retry $p.backup $p.install", script)


if __name__ == "__main__":
    unittest.main()
