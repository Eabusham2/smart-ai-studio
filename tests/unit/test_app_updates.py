"""Exercise real release parsing, streaming verification and ZIP staging offline."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from core import app_updates as updates


OLD = "1" * 40
NEW = "2" * 40


def encoded(value):
    return json.dumps(value).encode("utf-8")


class Fixture:
    def __init__(self, directory, target="Linux-x86_64", kind="portable", split=False, tag="latest"):
        self.target, self.tag, self.kind = target, tag, kind
        self.root_name = "SmartAI.app" if target.startswith("macOS") else "SmartAI-Windows" if target.startswith("Windows") else "SmartAI-Linux"
        self.code_relative = "Contents/Resources/app" if target.startswith("macOS") else "." if target.startswith("Windows") else "SmartAI.AppDir/usr/bin"
        self.install = Path(directory) / self.root_name
        self.code = self.install / self.code_relative
        self.code.mkdir(parents=True)
        self.old_build = {"schema": 1, "repository": updates.REPOSITORY, "commit": OLD,
                          "tag": tag, "target": target, "package_kind": kind,
                          "managed_paths": [updates.BUILD_FILE, "runtime", "core", "bootstrap_runtime.py"],
                          "install_managed_paths": ["Contents", "SmartAI.AppDir"]}
        (self.code / updates.BUILD_FILE).write_bytes(encoded(self.old_build))
        (self.code / "data").mkdir()
        (self.code / "data/memory.db").write_bytes(b"user memories")
        build = dict(self.old_build, commit=NEW)
        self.archive = f"SmartAI-{target}-{'portable' if kind == 'portable' else 'Setup'}.zip"
        prefix = self.root_name + "/" + (self.code_relative + "/" if self.code_relative != "." else "")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr(prefix + updates.BUILD_FILE, encoded(build))
            zipped.writestr(prefix + "bootstrap_runtime.py", "# new bootstrap\n")
            zipped.writestr(prefix + "core/example.py", "# new code\n")
            info = zipfile.ZipInfo(prefix + ("runtime/python/python.exe" if target.startswith("Windows") else "runtime/python/bin/python3"))
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            zipped.writestr(info, b"fixture interpreter")
        self.archive_bytes = stream.getvalue()
        self.assets, self.rows, self.responses = [], [], {}
        if split:
            middle = len(self.archive_bytes) // 2
            for index, part in enumerate((self.archive_bytes[:middle], self.archive_bytes[middle:])):
                self.add_asset(self.archive + f".part{index:03d}", part)
            parts = list(self.rows)
            self.add_asset(self.archive + ".parts.json", encoded({
                "archive": self.archive, "sha256": hashlib.sha256(self.archive_bytes).hexdigest(), "parts": parts}))
        else:
            self.add_asset(self.archive, self.archive_bytes)
        self.receipt_name = f"SmartAI-{target}.receipt.json"
        self.receipt = {"schema": 1, "target": target, "commit": NEW, "files": list(self.rows)}
        self.add_asset(self.receipt_name, encoded(self.receipt), recorded=False)
        self.release = {"tag_name": tag, "draft": False, "prerelease": False, "assets": self.assets}
        self.responses[updates.API_ROOT + "/releases/latest"] = encoded(self.release)
        self.responses[updates.API_ROOT + "/git/ref/tags/" + tag] = encoded({"object": {"type": "commit", "sha": NEW}})
        self.responses[updates.API_ROOT + "/compare/" + OLD + "..." + NEW] = encoded({"status": "ahead"})
        self.client = updates.AppUpdater(self.code)
        self.opened = []

        def open_response(url):
            self.opened.append(url)
            return io.BytesIO(self.responses[url])

        self.client._open = open_response

    def url(self, name):
        return f"https://github.com/{updates.REPOSITORY}/releases/download/{self.tag}/{name}"

    def add_asset(self, name, body, recorded=True):
        digest = hashlib.sha256(body).hexdigest()
        self.assets.append({"name": name, "size": len(body), "browser_download_url": self.url(name), "digest": "sha256:" + digest})
        if recorded:
            self.rows.append({"name": name, "size": len(body), "sha256": digest})
        self.responses[self.url(name)] = body

    def rewrite_release(self):
        self.responses[updates.API_ROOT + "/releases/latest"] = encoded(self.release)

    def rewrite_receipt(self):
        body = encoded(self.receipt)
        self.responses[self.url(self.receipt_name)] = body
        asset = next(a for a in self.assets if a["name"] == self.receipt_name)
        asset.update(size=len(body), digest="sha256:" + hashlib.sha256(body).hexdigest())
        self.rewrite_release()


class AppUpdateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="smart updater tests '")
        self.addCleanup(self.directory.cleanup)
        self.target = patch.object(updates, "_target_name", return_value="Linux-x86_64")
        self.target.start()
        self.addCleanup(self.target.stop)

    def fixture(self, **kwargs):
        return Fixture(self.directory.name, **kwargs)

    def test_same_rolling_tag_new_commit_is_available(self):
        fixture = self.fixture()
        result = fixture.client.check()
        self.assertEqual(result.status, "available")
        self.assertEqual(result.release.tag, "latest")
        self.assertEqual(result.release.commit, NEW)
        self.assertEqual(result.release.total_bytes, len(fixture.archive_bytes))
        self.assertIn(OLD[:12], fixture.client.current_label)

    def test_same_commit_needs_no_download_or_ancestry_lookup(self):
        fixture = self.fixture()
        fixture.responses[updates.API_ROOT + "/git/ref/tags/latest"] = encoded({"object": {"type": "commit", "sha": OLD}})
        self.assertEqual(fixture.client.check().status, "current")
        self.assertEqual(len(fixture.opened), 2)

    def test_older_release_and_diverged_history_are_never_offered(self):
        fixture = self.fixture()
        for relation, expected in (("behind", "ahead"), ("identical", "ahead"), ("diverged", "unsupported")):
            with self.subTest(relation=relation):
                fixture.responses[updates.API_ROOT + "/compare/" + OLD + "..." + NEW] = encoded({"status": relation})
                result = fixture.client.check()
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.release)

    def test_version_tag_annotation_and_setup_flavor(self):
        fixture = self.fixture(kind="setup", tag="v3.0.0")
        tag_object = "3" * 40
        fixture.responses[updates.API_ROOT + "/git/ref/tags/v3.0.0"] = encoded({"object": {"type": "tag", "sha": tag_object}})
        fixture.responses[updates.API_ROOT + "/git/tags/" + tag_object] = encoded({"object": {"type": "commit", "sha": NEW}})
        result = fixture.client.check()
        self.assertEqual(result.release.package_kind, "setup")
        self.assertTrue(result.release.archive_name.endswith("-Setup.zip"))

    def test_prereleases_and_drafts_are_not_installed(self):
        fixture = self.fixture()
        for flag in ("draft", "prerelease"):
            with self.subTest(flag=flag):
                fixture.release[flag] = True
                fixture.rewrite_release()
                self.assertIsNone(fixture.client.check().release)
                fixture.release[flag] = False

    def test_receipt_and_tag_must_match_during_rolling_publication(self):
        fixture = self.fixture()
        fixture.receipt["commit"] = OLD
        fixture.rewrite_receipt()
        with self.assertRaisesRegex(updates.UpdateError, "publication is still in progress"):
            fixture.client.check()

    def test_asset_digest_and_receipt_must_agree(self):
        fixture = self.fixture()
        fixture.assets[0]["digest"] = "sha256:" + "f" * 64
        fixture.rewrite_release()
        with self.assertRaisesRegex(updates.UpdateError, "checksums disagree"):
            fixture.client.check()

    def test_metadata_digest_detects_corruption(self):
        fixture = self.fixture()
        fixture.responses[fixture.url(fixture.receipt_name)] = b"x" * len(fixture.responses[fixture.url(fixture.receipt_name)])
        with self.assertRaisesRegex(updates.UpdateError, "checksum"):
            fixture.client.check()

    def test_release_url_cannot_point_to_another_repository(self):
        fixture = self.fixture()
        fixture.assets[0]["browser_download_url"] = "https://github.com/somebody/another/releases/download/latest/app.zip"
        fixture.rewrite_release()
        with self.assertRaisesRegex(updates.UpdateError, "this GitHub repository"):
            fixture.client.check()

    def test_split_package_is_reassembled_verified_and_extracted(self):
        fixture = self.fixture(split=True)
        update = fixture.client.check().release
        self.assertEqual(len(update.downloads), 2)
        reports = []
        prepared = fixture.client.prepare(update, lambda *args: reports.append(args))
        new_code = prepared.staged_root / fixture.code_relative
        self.assertEqual(json.loads((new_code / updates.BUILD_FILE).read_text())["commit"], NEW)
        self.assertEqual((new_code / "core/example.py").read_text(), "# new code\n")
        self.assertEqual((fixture.code / "data/memory.db").read_bytes(), b"user memories")
        self.assertEqual(json.loads((fixture.code / updates.BUILD_FILE).read_text())["commit"], OLD)
        self.assertEqual(reports[-1], ("Update ready to install", len(fixture.archive_bytes), len(fixture.archive_bytes)))
        self.assertFalse((prepared.staging_dir / fixture.archive).exists())
        manifest_calls = [url for url in fixture.opened if url.endswith(".parts.json")]
        self.assertEqual(len(manifest_calls), 2)  # Initial check and refresh, never duplicate fetches.

    def test_each_platform_retains_expected_layout_and_executable_permissions(self):
        for target in ("Linux-x86_64", "Windows-x86_64", "macOS-arm64"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory, patch.object(updates, "_target_name", return_value=target):
                fixture = Fixture(directory, target=target)
                prepared = fixture.client.prepare(fixture.client.check().release)
                self.assertEqual(prepared.staged_root.name, fixture.root_name)
                code = prepared.staged_root / fixture.code_relative
                executable = code / ("runtime/python/python.exe" if target.startswith("Windows") else "runtime/python/bin/python3")
                self.assertTrue(executable.is_file())
                self.assertTrue(executable.stat().st_mode & stat.S_IXUSR)

    def test_corrupt_archive_leaves_installed_data_and_no_staging_debris(self):
        fixture = self.fixture()
        release = fixture.client.check().release
        fixture.responses[fixture.url(fixture.archive)] = b"X" * len(fixture.archive_bytes)
        with self.assertRaisesRegex(updates.UpdateError, "SHA-256"):
            fixture.client.prepare(release)
        self.assertEqual((fixture.code / "data/memory.db").read_bytes(), b"user memories")
        self.assertEqual(list(Path(self.directory.name).glob(".smartai-update-*")), [])

    def test_short_and_oversized_downloads_are_rejected(self):
        fixture = self.fixture()
        release = fixture.client.check().release
        for body in (fixture.archive_bytes[:-1], fixture.archive_bytes + b"extra"):
            with self.subTest(size=len(body)):
                fixture.responses[fixture.url(fixture.archive)] = body
                with self.assertRaises(updates.UpdateError):
                    fixture.client.prepare(release)
                self.assertEqual(list(Path(self.directory.name).glob(".smartai-update-*")), [])

    def test_changed_release_requires_new_check_before_download(self):
        fixture = self.fixture()
        release = fixture.client.check().release
        with patch.object(fixture.client, "check", return_value=updates.UpdateCheck("current", "Current")):
            with self.assertRaisesRegex(updates.UpdateError, "changed"):
                fixture.client.prepare(release)
        self.assertNotIn(fixture.url(fixture.archive), fixture.opened)

    def test_low_disk_space_refuses_before_downloading(self):
        fixture = self.fixture()
        release = fixture.client.check().release
        with patch.object(updates.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(updates.UpdateError, "free space"):
                fixture.client.prepare(release)
        self.assertNotIn(fixture.url(fixture.archive), fixture.opened)

    def test_git_checkout_is_never_replaced(self):
        fixture = self.fixture()
        (fixture.code / ".git").write_text("gitdir: fixture\n")
        with self.assertRaisesRegex(updates.UpdateError, "Use Git"):
            fixture.client.prepare(fixture.client.check().release)

    def test_source_install_and_wrong_architecture_are_explicitly_unsupported(self):
        fixture = self.fixture()
        with patch.object(updates, "_target_name", return_value="macOS-arm64"):
            self.assertEqual(fixture.client.check().status, "unsupported")
        (fixture.code / updates.BUILD_FILE).unlink()
        client = updates.AppUpdater(fixture.code)
        with patch.object(client, "_open", side_effect=AssertionError("Unexpected network")):
            self.assertEqual(client.check().status, "unsupported")

    def test_network_failure_is_actionable(self):
        with patch.object(updates.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(updates.UpdateError, "Check your connection"):
                updates.AppUpdater(self.directory.name)._open(updates.API_ROOT + "/releases/latest")

    def test_abandoned_staging_is_removed_without_touching_installed_data(self):
        fixture = self.fixture()
        prepared = fixture.client.prepare(fixture.client.check().release)
        fixture.client.discard(prepared)
        self.assertFalse(prepared.staging_dir.exists())
        self.assertEqual((fixture.code / "data/memory.db").read_bytes(), b"user memories")
        forged = updates.PreparedUpdate(prepared.release, fixture.install, fixture.code)
        with self.assertRaises(updates.UpdateError):
            fixture.client.discard(forged)
        self.assertTrue(fixture.install.exists())

    def test_linux_case_distinct_module_names_remain_supported(self):
        root = Path(self.directory.name)
        archive = root / "case-distinct.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("SmartAI-Linux/core/Foo.py", "UPPER")
            zipped.writestr("SmartAI-Linux/core/foo.py", "lower")
        result = updates.extract_update(archive, root / "case-output", "Linux-x86_64")
        self.assertEqual((result / "core/Foo.py").read_text(), "UPPER")
        self.assertEqual((result / "core/foo.py").read_text(), "lower")

    def test_archive_rejects_traversal_links_and_case_collisions_before_writing(self):
        root = Path(self.directory.name)
        scenarios = [
            [("SmartAI-Windows/../../outside", b"bad", None)],
            [("/absolute", b"bad", None)],
            [("SmartAI-Windows/path\\escape", b"bad", None)],
            [("SmartAI-Windows/runtime", b"../../outside", stat.S_IFLNK | 0o777)],
            [("SmartAI-Windows/core.py", b"one", None), ("SmartAI-Windows/CORE.py", b"two", None)],
        ]
        for index, members in enumerate(scenarios):
            with self.subTest(index=index):
                archive = root / f"malformed-{index}.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    for name, body, mode in members:
                        info = zipfile.ZipInfo(name)
                        if mode:
                            info.external_attr = mode << 16
                        stream.writestr(info, body)
                destination = root / f"extract-{index}"
                with self.assertRaises(updates.UpdateError):
                    updates.extract_update(archive, destination, "Windows-x86_64")
                self.assertFalse(destination.exists())
        self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
