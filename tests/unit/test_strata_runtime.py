"""Managed Strata safety/lifecycle contracts without model or engine downloads."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from types import ModuleType, SimpleNamespace
import zipfile

import psutil
import pytest

from core import strata_runtime as runtime_module
from core.strata_runtime import StrataCancelled, StrataRuntime, StrataRuntimeError, extract_terminal


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "get_portable_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime_module, "host_platform", lambda: "linux-x86_64")
    return StrataRuntime("q2_0")


def install_marker(runtime):
    for path in (runtime._python(), runtime.terminal / "strata.py", runtime.terminal / "setup.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    runtime_module._write_json(runtime.terminal / ".studio-verified.json", {
        "release": runtime_module.RELEASE, "version": runtime_module.CONTROL_VERSION,
        "platform": runtime.platform, "sha256": runtime.artifact["sha256"],
        "key_id": runtime_module.SIGNING_KEY_ID,
    })


def prepared_model(runtime):
    config = runtime.session / "state" / "strata" / "launch" / f"{runtime.preset}.json"
    weight = runtime.root / "models" / "weights.gguf"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"local test model placeholder")
    runtime_module._write_json(config, {
        "model_name": runtime.model_info["strata_api_model"], "args": ["--native", str(weight)],
    })
    return {
        "ok": True, "profile": runtime.profile, "config": str(config),
        "model_id": runtime.model_info["strata_api_model"],
        "plan": {"model": {"family": runtime.model_info["strata_family"],
                           "model": runtime.model_info["native_strata_model"]},
                 "download": {"text": "Already prepared"}},
    }


def ready_owner(runtime):
    return {"ok": True, "state": "ready", "profile": runtime.profile,
            "owner_id": "a" * 32, "instance_id": "b" * 32}


def health_result(runtime, **changes):
    return {"service": "strata", "loaded": True, "api_key": True,
            "model": runtime.model_info["strata_api_model"], "max_context": 32768, **changes}


def fake_lifecycle(runtime, monkeypatch, *, bad_owner=None, bad_health=None):
    install_marker(runtime)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "prepare" in command:
            return prepared_model(runtime)
        if "start" in command:
            return bad_owner if bad_owner is not None else ready_owner(runtime)
        if "stop" in command:
            return {"ok": True, "state": "stopped"}
        raise AssertionError(command)
    monkeypatch.setattr(runtime, "_run", run)
    def http(path):
        if path == "/health":
            return bad_health if bad_health is not None else health_result(runtime)
        return {"data": [{"id": runtime.model_info["strata_api_model"], "status": {"value": "loaded"}}]}
    monkeypatch.setattr(runtime, "_http_json", http)
    return calls


def test_idle_inspection_is_offline_and_creates_no_terminal(runtime, monkeypatch):
    monkeypatch.setattr(runtime_module, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *a, **kw: pytest.fail("process started"))
    assert not runtime.is_prepared()
    runtime.stop()
    assert not runtime.terminal.exists()


def test_selection_and_memory_cap_fail_before_start(runtime, monkeypatch):
    with pytest.raises(ValueError, match="conflicts"):
        StrataRuntime("q2_0", runtime_module.get_strata_preset("gyro-s"))
    with pytest.raises(ValueError, match="finite"):
        StrataRuntime("q2_0", {**runtime.model_info, "memory_limit_gb": float("inf")})
    monkeypatch.setattr(runtime, "_ensure_terminal", lambda *a: pytest.fail("installation started"))
    with pytest.raises(ValueError, match="context"):
        runtime.start(1)


@pytest.mark.parametrize("model_key", ["q2_0", "gyro-s"])
@pytest.mark.parametrize("machine", ["arm64", "aarch64"])
def test_linux_arm_rejects_incompatible_packs_before_any_download(tmp_path, monkeypatch, model_key, machine):
    monkeypatch.setattr(runtime_module, "get_portable_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: machine)
    monkeypatch.setattr(runtime_module, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *a, **kw: pytest.fail("process started"))
    with pytest.raises(StrataRuntimeError, match="do not run on Linux ARM64"):
        StrataRuntime(model_key)
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("kv_mode", ["int8", "k8v4", "q4_0"])
def test_metal_rejects_unsupported_kv_before_installation(runtime, monkeypatch, kv_mode):
    monkeypatch.setattr(runtime_module, "host_platform", lambda: "macos-arm64")
    monkeypatch.setattr(runtime_module, "urlopen", lambda *a, **kw: pytest.fail("network used"))
    monkeypatch.setattr(runtime_module.subprocess, "Popen", lambda *a, **kw: pytest.fail("process started"))
    with pytest.raises(StrataRuntimeError, match="Metal engine supports fp16 KV only"):
        StrataRuntime("q2_0", {**runtime.model_info, "strata_kv_mode": kv_mode})
    assert not runtime.root.exists()


def test_metal_default_fp16_remains_available(runtime, monkeypatch):
    monkeypatch.setattr(runtime_module, "host_platform", lambda: "macos-arm64")
    metal = StrataRuntime("q2_0")
    assert metal.model_info["strata_kv_mode"] == "fp16"
    assert metal.platform == "macos-arm64"
    assert not metal.root.exists()


def test_child_environment_is_private_and_never_changes_parent(runtime, monkeypatch):
    monkeypatch.setenv("STRATA_UPDATE_TRUSTED_KEYS", "/another/key.json")
    monkeypatch.setenv("STRATA_ROUTE_PRIOR", "1")
    monkeypatch.setenv("STRATA_PROFILE_DIR", "/another/profile")
    before = dict(os.environ)
    runtime._api_key = "private-key"
    env = runtime._env()
    assert "STRATA_UPDATE_TRUSTED_KEYS" not in env
    assert "STRATA_ROUTE_PRIOR" not in env
    assert env["STRATA_PROFILE_DIR"] == str(runtime.session / "profiles")
    assert env["STRATA_RUNTIME_DIR"] == str(runtime.session / "runtime")
    assert env["STRATA_SETTINGS_FILE"] == str(runtime.root / "settings.json")
    assert env["SMARTAI_STRATA_API_KEY"] == "private-key"
    assert env.get("HOME") == before.get("HOME")
    assert env.get("PATH") == before.get("PATH")
    assert os.environ == before


def make_tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, content in members:
            info = tarfile.TarInfo(name)
            info.mode = 0o755
            if kind == "file":
                body = content.encode()
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
            else:
                info.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                info.linkname = content
                archive.addfile(info)


@pytest.mark.parametrize("name,kind,content", [
    ("../outside", "file", "bad"), ("/absolute", "file", "bad"),
    ("C:/escape", "file", "bad"), ("dir\\escape", "file", "bad"),
    ("python", "symlink", "../../outside"), ("python", "hardlink", "../outside"),
    ("pipe", "fifo", ""),
])
def test_archive_rejects_paths_links_and_special_files(tmp_path, name, kind, content):
    archive = tmp_path / "bad.tar.gz"
    make_tar(archive, [(name, kind, content)])
    with pytest.raises(StrataRuntimeError):
        extract_terminal(archive, tmp_path / "out", lambda: None)
    assert not (tmp_path / "outside").exists()


def test_archive_supports_internal_python_symlink_and_permissions(tmp_path):
    archive = tmp_path / "good.tar.gz"
    make_tar(archive, [("python/bin/python3.12", "file", "runtime"),
                       ("python/bin/python3", "symlink", "python3.12")])
    extract_terminal(archive, tmp_path / "out", lambda: None)
    binary = tmp_path / "out/python/bin/python3"
    assert binary.read_text() == "runtime"
    assert binary.is_symlink()
    assert os.readlink(binary) == "python3.12"
    mode = binary.stat().st_mode
    assert mode & 0o400 and mode & 0o200
    # Windows chmod only controls the read-only flag; it does not implement
    # POSIX execute bits for the extensionless Unix interpreter in this fixture.
    if os.name != "nt":
        assert mode & 0o100


def test_archive_supports_native_windows_zip_layout(tmp_path):
    archive = tmp_path / "terminal.zip"
    files = {
        "strata.py": b"# terminal entry point\n",
        "setup.py": b"# native preparation\n",
        "python/python.exe": b"MZ fixture interpreter; never executed",
        "bin/strata.cmd": b"@echo off\r\n",
    }
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in files.items():
            entry = zipfile.ZipInfo(name)
            entry.external_attr = 0o100755 << 16
            output.writestr(entry, content)
    destination = tmp_path / "out"
    extract_terminal(archive, destination, lambda: None)
    windows = StrataRuntime.__new__(StrataRuntime)
    windows.platform = "windows-x86_64"
    binary = windows._python(destination)
    assert binary == destination / "python/python.exe"
    for name, content in files.items():
        assert (destination / name).read_bytes() == content
        assert not (destination / name).is_symlink()
    # The Windows package needs ordinary writable files, not symlink privileges.
    with binary.open("ab") as stream:
        stream.write(b"\n")
    assert binary.read_bytes() == files["python/python.exe"] + b"\n"


def test_archive_rejects_duplicate_paths_and_zip_symlinks(tmp_path):
    archive = tmp_path / "duplicate.tar.gz"
    make_tar(archive, [("same", "file", "one"), ("same", "file", "two")])
    with pytest.raises(StrataRuntimeError, match="duplicate"):
        extract_terminal(archive, tmp_path / "out", lambda: None)
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as output:
        link = zipfile.ZipInfo("link")
        link.external_attr = 0o120777 << 16
        output.writestr(link, "../../outside")
    with pytest.raises(StrataRuntimeError, match="links"):
        extract_terminal(archive, tmp_path / "zip-out", lambda: None)


def test_archive_extraction_can_be_cancelled(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    make_tar(archive, [("one", "file", "one"), ("two", "file", "two")])
    def cancelled():
        raise StrataCancelled("cancelled")
    with pytest.raises(StrataCancelled):
        extract_terminal(archive, tmp_path / "out", cancelled)
    assert not (tmp_path / "out/one").exists()


class Download(io.BytesIO):
    def geturl(self):
        return "https://release-assets.githubusercontent.com/asset"


def test_download_checks_hash_and_length_before_any_execution(runtime, monkeypatch, tmp_path):
    payload = b"trusted release archive"
    monkeypatch.setattr(runtime_module, "urlopen", lambda *a, **kw: Download(payload))
    monkeypatch.setattr(runtime, "_run", lambda *a, **kw: pytest.fail("unverified binary executed"))
    with pytest.raises(StrataRuntimeError, match="checksum"):
        runtime._download("https://github.com/asset", tmp_path / "bad", len(payload), None, expected_sha="0" * 64)
    with pytest.raises(StrataRuntimeError, match="pinned size"):
        runtime._download("https://github.com/asset", tmp_path / "large", len(payload) - 1, None)
    output = tmp_path / "good"
    runtime._download("https://github.com/asset", output, len(payload), None, expected_sha=hashlib.sha256(payload).hexdigest())
    assert output.read_bytes() == payload


@pytest.mark.parametrize("verified", [False, True])
def test_install_publishes_only_after_signed_feed_authorizes_archive(runtime, monkeypatch, tmp_path, verified):
    archive = tmp_path / "fixture.tar.gz"
    make_tar(archive, [("terminal/python/bin/python3", "file", "binary"),
                       ("terminal/strata.py", "file", "entry"), ("terminal/setup.py", "file", "setup")])
    payload = archive.read_bytes()
    runtime.artifact = {"file": "fixture.tar.gz", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    def download(request, **kw):
        return Download(payload if request.full_url.endswith("fixture.tar.gz") else b"{}")
    monkeypatch.setattr(runtime_module, "urlopen", download)
    def signature_check(command, **kwargs):
        assert command[command.index("verify") + 3].startswith(runtime_module.RELEASE_URL)
        assert "KjnXZB0T1r3dmBNot8h+c4rTicRtllg1LH9fUlTw4ZY=" in command[2]
        assert not runtime.terminal.exists()
        if not verified:
            raise StrataRuntimeError("bad signature")
        return {"ok": True, "key_id": runtime_module.SIGNING_KEY_ID,
                "version": runtime_module.CONTROL_VERSION, "sha256": runtime.artifact["sha256"],
                "size": runtime.artifact["size"], "sequence": 117}
    monkeypatch.setattr(runtime, "_run", signature_check)
    if verified:
        runtime._ensure_terminal(None, None)
        assert runtime._installed()
        assert runtime._python().read_text() == "binary"
    else:
        with pytest.raises(StrataRuntimeError, match="signature"):
            runtime._ensure_terminal(None, None)
        assert not runtime.terminal.exists()
    assert not list(runtime.terminal.parent.glob(".install-*"))


def test_launch_has_explicit_parent_ownership_and_scoped_stop(runtime, monkeypatch):
    calls = fake_lifecycle(runtime, monkeypatch)
    result = runtime.start(32768)
    assert result["model_id"] == "qwen3.8-flash-next-q2_0"
    assert result["n_ctx"] == 32768
    assert result["base_url"].startswith("http://127.0.0.1:")
    assert len(result["api_key"]) >= 32
    assert runtime.is_prepared()
    prepare, start = calls[0][0], calls[1][0]
    assert "--restart" not in prepare and "--move-data" not in prepare
    assert prepare[prepare.index("--context") + 1] == "32768"
    assert prepare[prepare.index("--kv") + 1] == "fp16"
    assert prepare[prepare.index("--tune") + 1] == "skip"
    assert "--api-key-env" in prepare and result["api_key"] not in prepare
    assert start[start.index("--profile") + 1] == runtime.profile
    assert start[start.index("--die-with-pid") + 1] == str(os.getpid())
    runtime.stop()
    stop = calls[-1][0]
    assert stop[stop.index("--profile") + 1] == runtime.profile
    assert stop[stop.index("--if-owner-id") + 1] == "a" * 32
    assert stop[stop.index("--if-instance-id") + 1] == "b" * 32
    assert runtime.is_prepared()  # Unload never deletes the downloaded/prepared model.


@pytest.mark.parametrize("bad_health", [
    {"loaded": False}, {"service": "another-service"}, {"model": "another-model"},
    {"api_key": False}, {"max_context": 8192},
])
def test_exit_zero_is_not_loaded_and_failure_stops_owned_server(runtime, monkeypatch, bad_health):
    calls = fake_lifecycle(runtime, monkeypatch, bad_health=health_result(runtime, **bad_health))
    with pytest.raises(StrataRuntimeError):
        runtime.start(32768)
    assert "stop" in calls[-1][0]
    assert runtime._result is None


def test_preparation_only_success_cannot_be_reported_as_loaded(runtime, monkeypatch):
    calls = fake_lifecycle(runtime, monkeypatch, bad_owner={"ok": True, "serving": False})
    with pytest.raises(StrataRuntimeError, match="did not become ready"):
        runtime.start(32768)
    assert "stop" in calls[-1][0]


def test_wrong_catalog_model_and_removed_weights_invalidate_readiness(runtime, monkeypatch):
    install_marker(runtime)
    prepared = prepared_model(runtime)
    runtime._remember_prepared(prepared)
    assert runtime.is_prepared()
    (runtime.root / "models/weights.gguf").write_bytes(b"")
    assert not runtime.is_prepared()
    prepared["plan"]["model"]["model"] = "Gyro-S"
    with pytest.raises(StrataRuntimeError, match="different model"):
        runtime._remember_prepared(prepared)


def test_context_change_does_not_stop_a_working_server(runtime, monkeypatch):
    calls = fake_lifecycle(runtime, monkeypatch)
    runtime.start(32768)
    with pytest.raises(StrataRuntimeError, match="Unload"):
        runtime.start(65536)
    assert not any("stop" in command for command, _ in calls)
    assert runtime._result is not None


def test_cancel_after_preparation_prevents_native_server_start(runtime, monkeypatch):
    install_marker(runtime)
    cancel = threading.Event()
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        result = prepared_model(runtime)
        cancel.set()
        return result
    monkeypatch.setattr(runtime, "_run", run)
    with pytest.raises(StrataCancelled):
        runtime.start(32768, cancelled=cancel.is_set)
    assert len(commands) == 1 and "prepare" in commands[0]


def test_stop_during_installation_is_preserved_and_later_explicit_start_works(runtime, monkeypatch):
    fake_lifecycle(runtime, monkeypatch)
    entered = threading.Event()
    failures = []
    def blocked_install(progress, cancelled):
        entered.set()
        assert runtime._cancel.wait(3)
        runtime._check(cancelled)
    monkeypatch.setattr(runtime, "_ensure_terminal", blocked_install)
    def start():
        try:
            runtime.start(32768)
        except BaseException as error:
            failures.append(error)
    loader = threading.Thread(target=start)
    loader.start()
    assert entered.wait(3)
    runtime.stop()
    loader.join(3)
    assert not loader.is_alive() and isinstance(failures[0], StrataCancelled)
    monkeypatch.setattr(runtime, "_ensure_terminal", lambda *args: None)
    assert runtime.start(32768)["health"]["loaded"] is True
    runtime.stop()


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_detached_owner_pid_requires_profile_parent_executable_and_creation_time(runtime, monkeypatch, wrong_owner):
    runtime._owner_requested = True
    runtime._launch_epoch = time.time()
    runtime_module._write_json(runtime.session / "runtime" / f"{runtime.profile}.json", {
        "profile": runtime.profile, "host_pid": 4242,
    })
    class Process:
        pid = 4242
        def __init__(self, pid):
            assert pid == 4242
        def cmdline(self):
            return [str(runtime._python()), "-m", "strata_control.runtime", "--host", runtime.profile,
                    "--die-with-pid", str(os.getpid() + int(wrong_owner))]
        def exe(self):
            return str(runtime._python())
        def create_time(self):
            return runtime._launch_epoch
        def children(self, recursive=False):
            return []
        def status(self):
            return psutil.STATUS_RUNNING
    monkeypatch.setattr(psutil, "Process", Process)
    assert runtime.owned_pids() == ([] if wrong_owner else [4242])


@pytest.mark.parametrize("host", ["linux", "darwin"])
def test_preparation_adapter_calls_official_plan_then_prepare_without_start(runtime, monkeypatch, capsys, host):
    """Exercise the adapter's executable Python, with the official boundary replaced."""
    calls = []
    prepared = prepared_model(runtime)
    cfg = runtime_module._read_json(Path(prepared["config"]))
    settings = SimpleNamespace(name=runtime.preset, mtp="off")
    args = SimpleNamespace(feed="pinned-feed")
    class Runner:
        engine_problem = None
        def __init__(self, chosen, **kwargs):
            assert chosen is settings
            self.s = chosen
            assert kwargs["assume_yes"] is False
            assert kwargs["feed"] == "pinned-feed"
        def setup_args(self):
            # The pinned official Runner removes --mtp on Mac before invoking
            # setup, which would otherwise download an unused draft layer.
            return ["--yes", "--no-start"] + ([] if host == "darwin" else ["--mtp", "off"])
        def plan(self):
            calls.append("plan")
            return prepared["plan"]
        def run(self, *, start):
            assert start is False
            forwarded = self.setup_args()
            assert forwarded.count("--mtp") == 1
            assert forwarded[forwarded.index("--mtp") + 1] == "off"
            calls.append("prepare")
            return SimpleNamespace(serving=True, config=prepared["config"], profile=runtime.profile, notices=[])
        def _stop_children(self):
            calls.append("cleanup")
    def module(name, **values):
        result = ModuleType(name)
        result.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, result)
        return result
    module("strata_cli", PRODUCT_VERSION="0.2.8", version=lambda: "0.1.54")
    module("strata_cli.cli", parser=lambda: SimpleNamespace(parse_args=lambda _: args))
    cmd = SimpleNamespace(settings_from_args=lambda a, c: (settings, []), apply_defaults=lambda s, c: (s, []))
    store = SimpleNamespace(save=lambda s, n: calls.append("save"))
    module("strata_cli.launch", cmd=cmd, store=store)
    module("strata_cli.launch.catalog", Catalog=lambda: object())
    module("strata_cli.launch.runner", Runner=Runner, read_config=lambda _: cfg,
           files_present=lambda _: [], LaunchError=RuntimeError)
    module("strata_control", procs=SimpleNamespace(start_token=lambda _: "parent-start-token"))
    monkeypatch.setenv("STRATA_COMPONENTS_DIR", str(runtime.root / "components"))
    monkeypatch.setenv("SMARTAI_STRATA_OWNER_PID", str(os.getpid()))
    monkeypatch.setattr(sys, "argv", ["adapter", str(runtime.terminal), "prepare", "launch"])
    monkeypatch.setattr(sys, "platform", host)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The adapter installs SIGTERM/SIGINT handlers in its own subprocess. Do not
    # alter the pytest process's signal handlers while exercising that boundary.
    import signal
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    exec(compile(runtime_module._ADAPTER, "strata-adapter", "exec"), {})
    output = [json.loads(line[len(runtime_module._EVENT_PREFIX):]) for line in capsys.readouterr().out.splitlines()]
    assert calls == ["plan", "prepare", "save"]
    assert output[-1]["ok"] is True and output[-1]["model_id"] == cfg["model_name"]


def test_failed_owned_stop_retains_identity_until_cleanup_succeeds(runtime, monkeypatch):
    runtime._owner_requested = True
    runtime._owner = {"owner_id": "owned-instance", "instance_id": "owned-engine"}
    runtime._result = {"loaded": True}
    runtime._owned = {4242: 123.0}
    alive = True
    actions = []
    class Process:
        pid = 4242
        def create_time(self):
            return 123.0
        def status(self):
            return psutil.STATUS_RUNNING
        def is_running(self):
            return alive
        def terminate(self):
            actions.append("terminate")
        def kill(self):
            actions.append("kill")
    process = Process()
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, **kw: ([], list(processes)))
    monkeypatch.setattr(runtime, "owned_pids", lambda: [4242] if alive else [])
    commands = []
    def stop(command, **kwargs):
        commands.append(command)
        return {"ok": not alive}
    monkeypatch.setattr(runtime, "_run", stop)
    with pytest.raises(StrataRuntimeError, match="identity is retained"):
        runtime.stop()
    assert actions == ["terminate", "kill"]
    assert runtime._owner_requested is True
    assert runtime._owner == {"owner_id": "owned-instance", "instance_id": "owned-engine"}
    assert runtime._result == {"loaded": True}
    alive = False
    runtime.stop()
    assert runtime._owner_requested is False
    assert runtime._owner == {} and runtime._result is None
    assert all("--if-owner-id" in command and "--if-instance-id" in command for command in commands)


def test_real_subprocess_cancellation_is_bounded_and_does_not_touch_other_process(runtime, tmp_path):
    runtime.terminal.mkdir(parents=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    cancel, started = threading.Event(), threading.Event()
    failures = []
    script = 'import json,time; print("SMARTAI_STRATA " + json.dumps({"event":"progress","message":"ready"}),flush=True); time.sleep(30)'
    def worker():
        try:
            runtime._run([sys.executable, "-c", script], timeout=30,
                         progress=lambda _: started.set(), cancelled=cancel.is_set)
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=worker)
    try:
        thread.start()
        assert started.wait(5)
        pids = runtime.owned_pids()
        assert pids and unrelated.pid not in pids and os.getpid() not in pids
        cancel.set()
        thread.join(6)
        assert not thread.is_alive()
        assert isinstance(failures[0], StrataCancelled)
        assert unrelated.poll() is None
        assert all(not psutil.pid_exists(pid) for pid in pids)
    finally:
        cancel.set()
        unrelated.terminate()
        unrelated.wait(timeout=3)
        thread.join(6)


def test_real_subprocess_timeout_and_noisy_output_are_bounded(runtime):
    runtime.terminal.mkdir(parents=True)
    script = 'import time; print("x" * 10000, flush=True); time.sleep(20)'
    with pytest.raises(StrataRuntimeError, match="timed out"):
        runtime._run([sys.executable, "-c", script], timeout=0.2)
    assert not runtime.owned_pids()


def test_memory_limit_counts_studio_and_owned_child(runtime, monkeypatch):
    runtime.speed_memory_limit_gb = 1
    monkeypatch.setattr(runtime, "owned_pids", lambda: [4242, 4242])
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def memory_info(self):
            return SimpleNamespace(rss=700 * 1024**2)
    monkeypatch.setattr(psutil, "Process", Process)
    with pytest.raises(StrataRuntimeError, match="RAM limit"):
        runtime._check()


def test_http_verification_uses_authenticated_loopback_without_proxies(runtime, monkeypatch):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            payload = health_result(runtime) if self.path == "/health" else {
                "data": [{"id": runtime.model_info["strata_api_model"], "status": {"value": "loaded"}}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    runtime._port, runtime._api_key = server.server_port, "owned-key"
    try:
        health, context = runtime._validate_health(runtime.model_info["strata_api_model"], 32768)
        assert health["loaded"] is True and context == 32768
        assert seen == [("/health", "Bearer owned-key"), ("/v1/models", "Bearer owned-key")]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
