"""Studio-owned Strata Terminal, model preparation and local server lifecycle.

The official v0.2.8 archives are pinned by hash before any bundled code runs.
The bundled verifier then checks the release feed against the public key pinned
here. No installer script, shell profile, global Strata state, or global stop is
used. The small preparation adapter calls that exact release's launch Runner;
the server itself uses its documented, parent-bound ``runtime start`` command.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import secrets
import socket
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
import zipfile

from config.paths import get_portable_data_dir
from config.strata_models import (
    get_strata_preset, strata_launch_options, strata_memory_policy,
    validate_strata_selection,
)


RELEASE = "v0.2.8"
CONTROL_VERSION = "0.1.54"
RELEASE_URL = f"https://github.com/Niko1221/Strata/releases/download/{RELEASE}"
SIGNING_KEY_ID = "5d1bc23abbeffd0d"
SIGNING_PUBLIC_KEY = "KjnXZB0T1r3dmBNot8h+c4rTicRtllg1LH9fUlTw4ZY="
# Official install-manifest.txt and signed control packages, release v0.2.8.
ARTIFACTS = {
    "linux-x86_64": {
        "file": "strata-control-0.1.54-linux-x86_64.tar.gz", "size": 99_467_792,
        "sha256": "572520fbb0df30cc164fc7ce195c84908b02aaa67b569d7b1982d94cd18a364b",
    },
    "macos-arm64": {
        "file": "strata-control-0.1.54-macos-arm64.tar.gz", "size": 50_735_313,
        "sha256": "77b098d432cd16d34d8d0ce344d37dd92f94b5092bff73563991d07cc158e88f",
    },
    "windows-x86_64": {
        "file": "strata-control-0.1.54-windows-x86_64.zip", "size": 73_086_664,
        "sha256": "f24b7626b0b002ed69f092713bce05396b996562411b3c763fdc1be6ffa0ab22",
    },
}
_EVENT_PREFIX = "SMARTAI_STRATA "
_MAX_JSON = 2 * 1024 * 1024


class StrataRuntimeError(RuntimeError):
    """Installation, preparation or an owned server could not complete."""


class StrataCancelled(StrataRuntimeError):
    """The caller stopped a managed operation."""


def host_platform() -> str:
    system, machine = platform.system(), platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    key = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(system, system.lower())
    if key == "macos" and machine == "aarch64":
        machine = "arm64"
    result = f"{key}-{machine}"
    if result == "linux-aarch64":
        raise StrataRuntimeError(
            "This managed Strata integration does not support Linux ARM64, including DGX Spark. "
            "Use the model's native backend. No model download was started."
        )
    if result not in ARTIFACTS:
        raise StrataRuntimeError(
            "This computer has no supported pinned Strata Terminal. Use Apple Silicon, "
            "or Windows/Linux x86-64 with a supported GPU."
        )
    return result


def _read_json(path: Path) -> dict:
    try:
        if path.stat().st_size > _MAX_JSON:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise StrataRuntimeError(f"Strata's private directory is not a directory: {path}")
    if os.name != "nt":
        path.chmod(0o700)


def _write_json(path: Path, value: dict) -> None:
    _private_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=".studio-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _archive_path(name: str) -> Path:
    # Reject Windows drives/streams and separators even when inspecting on POSIX.
    parts = PurePosixPath(name).parts
    if not name or "\\" in name or ":" in name or name.startswith("/") or ".." in parts:
        raise StrataRuntimeError("The Strata archive contains an unsafe path.")
    return Path(*parts)


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def extract_terminal(archive: Path, destination: Path, check: Callable[[], None]) -> None:
    """Extract files first, then internal links; never follow archive links on write."""
    _private_dir(destination)
    limit = 1024 * 1024 * 1024  # The largest pinned Control expands to under 340 MB.
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > 50_000 or sum(m.file_size for m in members) > limit:
                raise StrataRuntimeError("The Strata archive exceeds its extraction limit.")
            seen = set()
            for member in members:
                relative = _archive_path(member.filename)
                if relative in seen or stat.S_ISLNK(member.external_attr >> 16):
                    raise StrataRuntimeError("The Strata ZIP contains duplicate paths or links.")
                seen.add(relative)
            for member in members:
                check()
                target = destination / _archive_path(member.filename)
                if member.is_dir():
                    _private_dir(target)
                    continue
                _private_dir(target.parent)
                with bundle.open(member) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        check()
                        output.write(chunk)
                target.chmod((member.external_attr >> 16 & 0o755) or 0o600)
        return
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 50_000 or sum(m.size for m in members) > limit:
            raise StrataRuntimeError("The Strata archive exceeds its extraction limit.")
        seen, links = set(), []
        for member in members:
            relative = _archive_path(member.name)
            if relative in seen or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise StrataRuntimeError("The Strata archive contains duplicate paths or special files.")
            seen.add(relative)
            if member.issym() or member.islnk():
                if "\\" in member.linkname or ":" in member.linkname or member.linkname.startswith("/"):
                    raise StrataRuntimeError("The Strata archive contains an unsafe link.")
                parent = (destination / relative).parent if member.issym() else destination
                if not _inside(parent / member.linkname, destination):
                    raise StrataRuntimeError("The Strata archive link escapes its directory.")
                links.append(member)
        for member in members:
            check()
            if member.issym() or member.islnk():
                continue
            target = destination / _archive_path(member.name)
            if member.isdir():
                _private_dir(target)
                continue
            _private_dir(target.parent)
            with bundle.extractfile(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    check()
                    output.write(chunk)
            target.chmod(member.mode & 0o755 or 0o600)
        for member in links:
            check()
            target = destination / _archive_path(member.name)
            _private_dir(target.parent)
            source = target.parent / member.linkname if member.issym() else destination / member.linkname
            # Recheck after earlier links were created, including chains.
            if not _inside(source, destination) or not _inside(target.parent, destination):
                raise StrataRuntimeError("The Strata archive contains an unsafe link chain.")
            if member.issym():
                target.symlink_to(member.linkname)
            else:
                if not source.is_file():
                    raise StrataRuntimeError("The Strata archive has an invalid hard link.")
                os.link(source, target)


# Shared by the two small, version-pinned subprocess adapters. The source list
# is written by Studio in its private session, never supplied as executable code.
_SOURCE_CHECKS = r'''
import json, os, stat
from pathlib import Path
def source_key(path):
    return os.path.normcase(str(Path(path).resolve()))
def checked_source(record):
    files = record.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 256:
        raise RuntimeError("The local Strata selection has no complete file identity")
    seen = set()
    for item in files:
        path = Path(item["path"])
        info = path.stat()
        key = source_key(path)
        if (key in seen or not stat.S_ISREG(info.st_mode)
                or info.st_size != item["size"] or info.st_mtime_ns != item["mtime_ns"]
                or ("ctime_ns" in item and info.st_ctime_ns != item["ctime_ns"])
                or ("realpath" in item and key != source_key(item["realpath"]))):
            raise RuntimeError("The chosen model files changed. Import the current files again before loading.")
        seen.add(key)
    return seen
def inspect_source(record, catalog):
    from strata_control import models
    expected = checked_source(record)
    preview = models.plan_import(str(record["path"]))
    candidate = preview.get("candidate_selection") or {}
    if (preview.get("issues") or preview.get("file_set_status") != "plausible"
            or not candidate.get("family") or not candidate.get("model")):
        words = preview.get("plain_issues") or preview.get("issues") or ["unsupported model layout"]
        raise RuntimeError("Strata cannot prepare the selected files: " + "; ".join(str(x) for x in words))
    companions = {source_key(p) for p in record.get("companions", [])}
    found = {source_key(item["path"]) for item in preview.get("files") or []}
    if not expected - companions <= found or not found <= expected:
        raise RuntimeError("Strata's inspection selected different files from the chosen model")
    setup = catalog.strata.setup_module()
    family, model = candidate["family"], candidate["model"]
    definition = getattr(setup, "FAMILIES", {}).get(family, {})
    if definition.get("ready"):
        raise RuntimeError("This Strata recipe uses a separate ready-made pack and would discard the chosen GGUF. Use its native backend.")
    if family == "deepseek41":
        definition = getattr(setup, "MULTI_FILE_FAMILIES", {}).get(family, {})
        recipe = definition.get("models", {}).get(model, {})
        aliases = getattr(setup, "DSV41_LEGACY_NAMES", {})
        missing = []
        paths = [Path(item["path"]) for item in record["files"]]
        for name in recipe.get("shared", []):
            allowed = {name, aliases.get(name)}
            match = next((p for p in paths if p.name in allowed), None)
            pin = getattr(setup, "DSV41_ALL_FILES", {}).get(match.name) if match else None
            if match is None or not pin or match.stat().st_size != pin[0]:
                missing.append(name)
        if missing:
            raise RuntimeError("The local Strata model needs these companion files before loading: " + ", ".join(missing)
                               + ". Studio will not download replacement model weights.")
    elif not definition or family in getattr(setup, "OWN_PATH_FAMILIES", {}):
        raise RuntimeError("This Strata architecture has no supported local preparation path in this integration. Use its native backend.")
    result = dict(record)
    result["recipe"] = {"family": family, "model": model, "basis": candidate.get("basis")}
    result["messages"] = list(preview.get("messages") or [])
    return result
'''


_LOCAL_SETUP_ADAPTER = _SOURCE_CHECKS + r'''
import importlib.util, sys
root, record_path = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
import strata_cli
if strata_cli.PRODUCT_VERSION != "0.2.8" or strata_cli.version() != "0.1.54":
    raise RuntimeError("Local-model preparation requires the pinned Strata Terminal")
record = json.loads(record_path.read_text(encoding="utf-8"))
checked_source(record)
spec = importlib.util.spec_from_file_location("studio_strata_setup", root / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
recipe = record["recipe"]
companions = {source_key(p) for p in record.get("companions", [])}
shards = [Path(item["path"]) for item in record["files"] if source_key(item["path"]) not in companions]
family = getattr(setup, "FAMILIES", {}).get(recipe["family"], {})
if family.get("ready"):
    raise RuntimeError("A ready-made Strata pack cannot replace the chosen local model")
if recipe["family"] == "deepseek41":
    aliases = getattr(setup, "DSV41_LEGACY_NAMES", {})
    shared = [Path(item["path"]) for item in record["files"] if source_key(item["path"]) in companions]
    def exact_named(paths, name):
        checked_source(record)
        for allowed in (name, aliases.get(name)):
            match = next((path for path in paths if path.name == allowed), None)
            if match is not None:
                return match
        raise RuntimeError("A required DeepSeek file is not part of the selected local model: " + name)
    # DeepSeek has separate selectors: do not prefer a canonical neighbour over
    # a selected legacy shard or discover unselected companions in parent dirs.
    setup.dsv41_pick = lambda folder, name: exact_named(shards, name)
    setup.dsv41_find_shared = lambda args, name, folder, shared_dir: exact_named(shared, name)
else:
    if not family:
        raise RuntimeError("The selected local Strata preparation recipe is unavailable")
    def exact_shards(folder, selected_family, selected_model):
        checked_source(record)
        if selected_model != recipe["model"] or selected_family != family:
            raise RuntimeError("Strata setup changed the selected model recipe")
        return list(shards)
    # Official launch reduces --model-path FILE to --gguf-dir PARENT. Preserve
    # its exact inspected shard set even when a canonical neighbour is present.
    setup.gguf_dir_shards = exact_shards
def no_model_download(*args, **kwargs):
    raise RuntimeError("A required local model companion is missing. Studio will not download replacement model weights.")
setup.download = no_model_download
sys.argv = [str(root / "setup.py"), *sys.argv[3:]]
raise SystemExit(setup.main())
'''


# Runs only with the hash-verified bundled interpreter and source. Keeping this
# adapter out of the installed tree lets the official Terminal remain unedited.
_ADAPTER = _SOURCE_CHECKS + r'''
import base64, json, os, signal, sys, threading
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
import strata_cli
if strata_cli.PRODUCT_VERSION != "0.2.8" or strata_cli.version() != "0.1.54":
    raise RuntimeError("The Strata adapter requires Terminal v0.2.8 / Control 0.1.54")
def emit(value):
    print("SMARTAI_STRATA " + json.dumps(value, ensure_ascii=False), flush=True)
if mode == "verify":
    from strata_control import feed
    raw, sig = Path(sys.argv[3]).read_bytes(), Path(sys.argv[4]).read_bytes()
    trusted = {"5d1bc23abbeffd0d": base64.b64decode("KjnXZB0T1r3dmBNot8h+c4rTicRtllg1LH9fUlTw4ZY=")}
    verified = feed.verify_and_parse(raw, sig, source=sys.argv[5], trusted=trusted, channel="stable", last_sequence=117)
    package = next(p for p in verified.control if p.platform == sys.argv[6])
    emit({"event":"result", "ok":True, "key_id":verified.key_id, "sequence":verified.sequence,
          "version":package.version, "sha256":package.sha256, "size":package.size})
elif mode == "prepare":
    from strata_cli.cli import parser
    from strata_cli.launch import cmd, store
    from strata_cli.launch.catalog import Catalog
    from strata_cli.launch.runner import Runner, read_config, files_present, LaunchError
    from strata_control import procs
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    args = parser().parse_args(sys.argv[3:])
    catalog = Catalog()
    settings, notes = cmd.settings_from_args(args, catalog)
    settings, defaults = cmd.apply_defaults(settings, catalog)
    notes += defaults
    for note in notes:
        if note.level == "error":
            raise RuntimeError(note.line())
    local = None
    source_file = os.environ.get("SMARTAI_STRATA_SOURCE_FILE")
    if getattr(args, "model_path", None):
        if not source_file or getattr(args, "family", None) or getattr(args, "model", None):
            raise RuntimeError("A local Strata launch must preserve its chosen files without forcing a catalog model")
        local = inspect_source(json.loads(Path(source_file).read_text(encoding="utf-8")), catalog)
        if source_key(args.model_path) != source_key(local["path"]):
            raise RuntimeError("Strata launch changed the chosen source path")
        if local["recipe"]["family"] == "deepseek41":
            # This comes only from the successful exact-file preflight above.
            # DeepSeek's folder scan otherwise prefers the first neighbouring
            # variant in its catalog, even when --model-path names one file.
            settings.family = local["recipe"]["family"]
            settings.model = local["recipe"]["model"]
            settings.speed = "full"
            settings.kv = None  # this separate engine uses its native KV representation
            settings.extra = {**settings.extra, "--dspark": "none"}
            settings.extra.pop("--spec", None)  # the separate DeepSeek engine has no Qwen --spec flag
        Path(source_file).write_text(json.dumps(local), encoding="utf-8")
    def progress(event):
        emit({"event":"progress", "stage":event.stage, "message":event.text, "percent":event.pct})
    def unsupported(title, detail):
        raise LaunchError(detail, "A supported native engine is required before downloading this model.", 6)
    class StudioRunner(Runner):
        def deepseek_files(self):
            if local and local["recipe"]["family"] != "deepseek41":
                # A DeepSeek neighbour must not intercept an inspected Qwen file.
                return False
            return super().deepseek_files()
        def _spawn(self, argv, **kwargs):
            if local and len(argv) > 1 and source_key(argv[1]) == source_key(root / "setup.py"):
                checked_source(local)
                argv = [argv[0], str(Path(source_file).with_name("prepare-local.py")),
                        str(root), source_file, *argv[2:]]
            return super()._spawn(argv, **kwargs)
        def setup_args(self):
            forwarded = super().setup_args()
            # Control 0.1.54's Mac filter drops --mtp although mac_control_setup
            # accepts it. Keep our explicit choice so preparation does not fetch
            # an unused ~5 GB draft layer. The installed upstream stays intact.
            if sys.platform == "darwin" and self.s.mtp == "off" and "--mtp" not in forwarded:
                forwarded += ["--mtp", "off"]
            return forwarded
    runner = StudioRunner(settings, catalog=catalog, emit=progress, cancel=cancel, feed=args.feed,
                    components_dir=os.environ["STRATA_COMPONENTS_DIR"], assume_yes=False, confirm=unsupported)
    owner_pid = int(os.environ["SMARTAI_STRATA_OWNER_PID"])
    owner_token = procs.start_token(owner_pid)
    if owner_token is None:
        raise RuntimeError("Studio ended before Strata preparation began")
    finished = threading.Event()
    def follow_owner():
        while not finished.wait(0.25):
            if procs.start_token(owner_pid) != owner_token:
                cancel.set()
                runner._stop_children()
                return
    threading.Thread(target=follow_owner, name="studio-strata-owner", daemon=True).start()
    try:
        plan = runner.plan()
        if local:
            selected = plan.get("model") or {}
            recipe = local["recipe"]
            if (selected.get("family") != recipe["family"] or selected.get("model") != recipe["model"]
                    or selected.get("own_files") is not True):
                raise RuntimeError("Strata's launch plan substituted a different model")
            parts = (plan.get("download") or {}).get("parts") or []
            if any(p.get("kind") != "engine" and float(p.get("gb") or 0) > 0 for p in parts):
                raise RuntimeError("The chosen local model is incomplete; Strata would download additional model weights. "
                                   "Provide its companion files before loading.")
            checked_source(local)
            plan["source_identity"] = local
        plan["warnings"] = list(plan.get("warnings") or []) + [n.line() for n in notes]
        emit({"event":"plan", "plan":plan})
        result = runner.run(start=False)
    except BaseException:
        runner._stop_children()
        raise
    finally:
        finished.set()
    if runner.engine_problem or not result.serving:
        raise RuntimeError(runner.engine_problem or result.final_text or "No usable Strata engine")
    cfg = read_config(Path(result.config))
    if files_present(cfg):
        raise RuntimeError("Prepared Strata model files are missing")
    if not cfg.get("model_name"):
        raise RuntimeError("Prepared Strata model has no model identity")
    if local:
        checked_source(local)
        if local["recipe"]["family"] == "deepseek41":
            argv = cfg.get("args") or []
            def value(flag):
                return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
            if value("--dspark") != "none" or str(value("--experts")) != "6":
                raise RuntimeError("This DeepSeek engine did not retain full experts and disabled drafting")
    store.save(settings, settings.name)
    emit({"event":"result", "ok":True, "profile":result.profile, "config":result.config,
          "model_id":cfg["model_name"], "plan":plan, "source_identity":local, "notices":result.notices})
else:
    raise RuntimeError("Unknown Studio Strata adapter operation")
'''


class StrataRuntime:
    """One owned instance, with reusable private Terminal and prepared model data."""

    INSTALL_TIMEOUT = 15 * 60
    PREPARE_TIMEOUT = 6 * 60 * 60
    START_TIMEOUT = 1800
    STOP_TIMEOUT = 25

    def __init__(self, model_key: str, model_info: dict | None = None):
        self._local = (model_info or {}).get("source_kind") == "strata_local"
        if self._local:
            self.model_info = validate_strata_selection(model_info)
            if model_key != self.model_info["model_key"]:
                raise ValueError("The local Strata model key conflicts with its source identity.")
        else:
            chosen = get_strata_preset(model_key)
            self.model_info = validate_strata_selection(model_info or chosen)
            if self.model_info["model_key"] != chosen["model_key"]:
                raise ValueError("The Strata runtime model key conflicts with its model information.")
        self.model_key = self.model_info["model_key"]
        self.speed_memory_limit_gb = (model_info or {}).get("memory_limit_gb")
        strata_memory_policy(self.speed_memory_limit_gb)
        self.root = Path(get_portable_data_dir()).resolve() / "runtime" / "strata"
        self.platform = host_platform()
        if self.platform == "macos-arm64" and self.model_info["strata_kv_mode"] != "fp16":
            raise StrataRuntimeError(
                "The pinned Strata Metal engine supports fp16 KV only. Select fp16 before loading this model."
            )
        self.artifact = ARTIFACTS[self.platform]
        self.terminal = self.root / "terminal" / f"{CONTROL_VERSION}-{self.platform}"
        # Upstream keys prepared packs by recipe name, not by the user's source.
        # Give every local artifact its own install/discovery/data roots so two
        # same-layout fine-tunes never share tokenizer or expert-cache files.
        self._model_home = (self.root / "local-models" / self.model_info["strata_source_id"]
                            if self._local else self.root)
        self._data_dir = self._model_home / "data" if self._local else self.root / "models"
        self.preset = f"smartai-{uuid.uuid4().hex}"
        self.profile = f"launch-{self.preset}"
        self.session = self.root / "sessions" / self.preset
        self._cancel = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process = None
        self._owned: dict[int, float] = {}
        self._api_key = ""
        self._port = 0
        self._owner: dict[str, Any] = {}
        self._owner_requested = False
        self._launch_epoch = 0.0
        self._result: dict | None = None
        self._plan: dict = {}

    def _emit(self, progress, message: str) -> None:
        if progress is not None:
            try:
                progress(str(message).replace(self._api_key, "[redacted]") if self._api_key else str(message))
            except Exception:
                pass  # A status widget cannot invalidate a loaded model.

    def _check(self, cancelled=None) -> None:
        try:
            requested = cancelled is not None and bool(cancelled())
        except Exception:
            requested = True
        if self._cancel.is_set() or requested:
            raise StrataCancelled("Strata loading was cancelled; downloaded model data is kept.")
        if self.speed_memory_limit_gb is not None:
            import psutil
            limit = strata_memory_policy(self.speed_memory_limit_gb)["application_limit_gb"] * 1024**3
            rss = 0
            for pid in {os.getpid(), *self.owned_pids()}:
                try:
                    rss += psutil.Process(pid).memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if rss > limit:
                raise StrataRuntimeError("Strata loading exceeded Studio's application RAM limit; the owned runtime was stopped.")

    def _env(self, terminal: Path | None = None) -> dict[str, str]:
        env = dict(os.environ)
        # Ambient Strata knobs may alter quality, trust or someone else's state.
        for name in list(env):
            if name.startswith(("STRATA_", "SMARTAI_STRATA_")) or name in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}:
                env.pop(name, None)
        env.update({
            "STRATA_INSTALL_DIR": str(self._model_home / "install" if self._local else terminal or self.terminal),
            "STRATA_DATA_DIR": str(self._data_dir),
            "STRATA_SETTINGS_FILE": str(self._model_home / "settings.json"),
            "STRATA_DOCUMENTS_DIR": str(self._model_home / "documents"),
            "STRATA_PROFILE_DIR": str(self.session / "profiles"),
            "STRATA_RUNTIME_DIR": str(self.session / "runtime"),
            "STRATA_LAUNCH_PRESET_DIR": str(self.session / "launch-presets"),
            "STRATA_PRESET_DIR": str(self.session / "tuning-presets"),
            "STRATA_COMPONENTS_DIR": str(self.root / "components" / RELEASE),
            "STRATA_UPDATE_FEED": RELEASE_URL + "/" + self._feed_name(),
            "SMARTAI_STRATA_API_KEY": self._api_key,
            "SMARTAI_STRATA_OWNER_PID": str(os.getpid()),
            "XDG_STATE_HOME": str(self.session / "state"),
            "XDG_CACHE_HOME": str(self._model_home / "cache"),
            "XDG_DATA_HOME": str(self._model_home / "xdg-data" if self._local else self.root / "data"),
            "XDG_CONFIG_HOME": str(self._model_home / "config"),
            "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
        })
        if os.name == "nt":
            env["LOCALAPPDATA"] = str(self.session / "localappdata")
            env["APPDATA"] = str(self._model_home / "roaming")
        if self._local:
            env["SMARTAI_STRATA_SOURCE_FILE"] = str(self.session / "source.json")
        return env

    def _feed_name(self) -> str:
        return "strata-feed.json"

    def _python(self, terminal: Path | None = None) -> Path:
        return (terminal or self.terminal) / ("python/python.exe" if self.platform.startswith("windows-") else "python/bin/python3")

    def _installed(self) -> bool:
        marker = _read_json(self.terminal / ".studio-verified.json")
        return bool(
            marker.get("release") == RELEASE and marker.get("version") == CONTROL_VERSION
            and marker.get("platform") == self.platform and marker.get("sha256") == self.artifact["sha256"]
            and marker.get("key_id") == SIGNING_KEY_ID
            and self._python().is_file() and (self.terminal / "strata.py").is_file()
            and (self.terminal / "setup.py").is_file()
        )

    def _source_record(self) -> dict | None:
        if not self._local:
            return None
        current = validate_strata_selection(self.model_info)
        if current["strata_source_id"] != self.model_info["strata_source_id"]:
            raise StrataRuntimeError("The chosen model files changed. Import the current files again before loading.")
        companions = current.get("strata_companion_files") or []
        return {
            "id": current["strata_source_id"], "path": current["strata_model_path"],
            "files": current["strata_source_files"],
            "companions": [item["path"] if isinstance(item, dict) else str(item) for item in companions],
        }

    def _same_source(self, record: dict | None) -> bool:
        expected = self._source_record()
        return (isinstance(record, dict) and expected is not None
                and all(record.get(key) == value for key, value in expected.items()))

    def is_prepared(self) -> bool:
        """Cheap offline hint; start still asks upstream to check all prepared files."""
        if not self._installed():
            return False
        record = _read_json(self.root / "prepared" / f"{self.model_key}.json")
        if record.get("release") != RELEASE or record.get("model_key") != self.model_key:
            return False
        source_paths = set()
        if self._local:
            try:
                if not self._same_source(record.get("source_identity")):
                    return False
                source_paths = {Path(item["path"]).resolve() for item in self.model_info["strata_source_files"]}
            except (OSError, ValueError, StrataRuntimeError):
                return False
        config = Path(record.get("config") or "")
        if not _inside(config, self.root) or not config.is_file():
            return False
        cfg = _read_json(config)
        if cfg.get("model_name") != record.get("model_id") or not isinstance(cfg.get("args"), list):
            return False
        files = record.get("files")
        if not isinstance(files, list) or not files:
            return False
        try:
            return all(
                isinstance(item, dict) and isinstance(item.get("path"), str)
                and (_inside(Path(item["path"]), self.root) or Path(item["path"]).resolve() in source_paths)
                and Path(item["path"]).exists()
                and ("size" not in item or Path(item["path"]).stat().st_size == item["size"] > 0)
                for item in files
            )
        except OSError:
            return False

    @contextmanager
    def _install_lock(self, cancelled):
        _private_dir(self.root)
        with (self.root / ".install.lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
            else:
                import fcntl
            deadline = time.monotonic() + self.INSTALL_TIMEOUT
            while True:
                self._check(cancelled)
                try:
                    if os.name == "nt":
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise StrataRuntimeError("Another Studio process is still preparing Strata.") from None
                    self._cancel.wait(0.1)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _download(self, url: str, path: Path, maximum: int, cancelled, progress=None, expected_sha=None) -> None:
        self._check(cancelled)
        digest, size = hashlib.sha256(), 0
        deadline = time.monotonic() + self.INSTALL_TIMEOUT
        request = Request(url, headers={"User-Agent": "SmartAI-Studio-Strata/1.0"})
        with urlopen(request, timeout=10) as response, path.open("xb") as output:
            if urlsplit(response.geturl()).scheme != "https":
                raise StrataRuntimeError("Strata downloads require HTTPS.")
            while chunk := response.read(1024 * 1024):
                self._check(cancelled)
                if time.monotonic() > deadline:
                    raise StrataRuntimeError("The Strata Terminal download timed out.")
                size += len(chunk)
                if size > maximum:
                    raise StrataRuntimeError("The Strata download exceeds the pinned size.")
                digest.update(chunk)
                output.write(chunk)
                if progress:
                    self._emit(progress, f"Downloading verified Strata Terminal: {size / 1024**2:.0f} / {maximum / 1024**2:.0f} MiB")
        if expected_sha and (digest.hexdigest() != expected_sha or size != maximum):
            raise StrataRuntimeError("The Strata Terminal does not match the pinned release checksum and size.")

    def _ensure_terminal(self, progress, cancelled) -> None:
        with self._install_lock(cancelled):
            if self._installed():
                return
            _private_dir(self.terminal.parent)
            with tempfile.TemporaryDirectory(prefix=".install-", dir=self.terminal.parent) as folder:
                staging = Path(folder)
                archive, feed = staging / self.artifact["file"], staging / self._feed_name()
                signature = Path(str(feed) + ".sig")
                self._emit(progress, f"Installing official Strata {RELEASE} in Studio's runtime directory")
                self._download(RELEASE_URL + "/" + self.artifact["file"], archive, self.artifact["size"],
                               cancelled, progress, self.artifact["sha256"])
                self._download(RELEASE_URL + "/" + feed.name, feed, _MAX_JSON, cancelled)
                self._download(RELEASE_URL + "/" + signature.name, signature, 16 * 1024, cancelled)
                unpacked = staging / "unpacked"
                extract_terminal(archive, unpacked, lambda: self._check(cancelled))
                candidates = [unpacked, *[p for p in unpacked.iterdir() if p.is_dir() and not p.is_symlink()]]
                roots = [p for p in candidates if (p / "strata.py").is_file() and self._python(p).is_file()]
                if len(roots) != 1:
                    raise StrataRuntimeError("The official Strata archive has an unexpected Terminal layout.")
                terminal = roots[0]
                self._emit(progress, "Checking Strata's signed release feed")
                verified = self._run(
                    [str(self._python(terminal)), "-c", _ADAPTER, str(terminal), "verify", str(feed), str(signature),
                     RELEASE_URL + "/" + feed.name, self.platform], terminal=terminal, timeout=60, cancelled=cancelled,
                )
                if not (verified.get("ok") is True and verified.get("key_id") == SIGNING_KEY_ID
                        and verified.get("version") == CONTROL_VERSION
                        and verified.get("sha256") == self.artifact["sha256"]
                        and verified.get("size") == self.artifact["size"]):
                    raise StrataRuntimeError("Strata's signed feed does not authorize the pinned Terminal archive.")
                _write_json(terminal / ".studio-verified.json", {
                    "release": RELEASE, "version": CONTROL_VERSION, "platform": self.platform,
                    "sha256": self.artifact["sha256"], "key_id": SIGNING_KEY_ID,
                    "sequence": verified.get("sequence"),
                })
                self._check(cancelled)
                if self.terminal.exists():
                    raise StrataRuntimeError("A damaged Strata Terminal already occupies this version's directory; move that runtime directory aside and retry.")
                os.replace(terminal, self.terminal)

    def _capture_owner(self) -> None:
        if not self._owner_requested:
            return
        record = _read_json(self.session / "runtime" / f"{self.profile}.json")
        if record.get("profile") != self.profile or type(record.get("host_pid")) is not int:
            return
        import psutil
        try:
            process = psutil.Process(record["host_pid"])
            args = process.cmdline()
            def option(name):
                return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None
            if ("strata_control.runtime" not in args or option("--host") != self.profile
                    or option("--die-with-pid") != str(os.getpid())
                    or not _inside(Path(process.exe()), self.terminal)
                    or process.create_time() < self._launch_epoch - 1):
                return
            with self._process_lock:
                self._owned[process.pid] = process.create_time()
        except (psutil.Error, OSError, ValueError):
            return

    def owned_pids(self) -> list[int]:
        """Verified tracked processes and their descendants, for Studio's watchdog."""
        import psutil
        self._capture_owner()
        with self._process_lock:
            identities = dict(self._owned)
        found = {}
        for pid, started in identities.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() != started or process.status() == psutil.STATUS_ZOMBIE:
                    continue
                found[pid] = started
                for child in process.children(recursive=True):
                    found[child.pid] = child.create_time()
            except (psutil.Error, OSError):
                continue
        found.pop(os.getpid(), None)
        with self._process_lock:
            self._owned.update(found)
        return sorted(found)

    def _terminate_process(self, process) -> None:
        if process is None or process.poll() is not None:
            return
        import psutil
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        # Descendants may be setup/download workers; each Process retains its
        # creation time so psutil refuses a recycled PID when signalling it.
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        _, alive = psutil.wait_procs(children, timeout=1)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass

    def _run(self, command: list[str], *, terminal: Path | None = None, timeout: float,
             progress=None, cancelled=None, stopping=False) -> dict:
        terminal = terminal or self.terminal
        if not stopping:
            self._check(cancelled)
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(command, cwd=terminal, env=self._env(terminal), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **flags)
        import psutil
        with self._process_lock:
            self._process = process
            try:
                self._owned[process.pid] = psutil.Process(process.pid).create_time()
            except psutil.Error:
                pass
        lines: queue.Queue = queue.Queue(maxsize=8)
        def read():
            try:
                for line in iter(lambda: process.stdout.readline(_MAX_JSON + 1), b""):
                    try:
                        lines.put_nowait(line.decode("utf-8", "replace"))
                    except queue.Full:
                        try:
                            lines.get_nowait()
                        except queue.Empty:
                            pass
                        lines.put_nowait(line.decode("utf-8", "replace"))
            except (OSError, ValueError):
                pass
            finally:
                reader_done.set()
        reader_done = threading.Event()
        reader = threading.Thread(target=read, name="studio-strata-output", daemon=True)
        reader.start()
        deadline, result, tail = time.monotonic() + timeout, {}, ""
        next_ownership_check = 0.0
        try:
            while process.poll() is None or not reader_done.is_set() or not lines.empty():
                if not stopping:
                    self._check(cancelled)
                if time.monotonic() >= next_ownership_check:
                    self.owned_pids()
                    next_ownership_check = time.monotonic() + 0.5
                if time.monotonic() > deadline:
                    raise StrataRuntimeError("The managed Strata operation timed out.")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                tail = (tail + line)[-8192:]
                raw = line[len(_EVENT_PREFIX):] if line.startswith(_EVENT_PREFIX) else line
                try:
                    item = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("event") == "progress":
                    self._emit(progress, item.get("message") or "Preparing Strata")
                elif item.get("event") == "plan":
                    self._plan = item.get("plan") or {}
                    self._emit(progress, (self._plan.get("download") or {}).get("text") or "Strata checked this computer's model placement")
                else:
                    result = item
            if process.returncode != 0:
                safe = tail.replace(self._api_key, "[redacted]") if self._api_key else tail
                raise StrataRuntimeError(f"Strata exited with code {process.returncode}: {safe.strip()[-2500:]}")
            if not result:
                raise StrataRuntimeError("Strata returned no machine-readable result.")
            return result
        finally:
            self._terminate_process(process)
            reader.join(timeout=1)
            process.stdout.close()
            with self._process_lock:
                if self._process is process:
                    self._process = None

    def _cli(self, *args: str) -> list[str]:
        return [str(self._python()), str(self.terminal / "strata.py"), *args]

    def _remember_prepared(self, prepared: dict) -> None:
        config = Path(prepared.get("config") or "")
        if not _inside(config, self.session) or not config.is_file():
            raise StrataRuntimeError("Strata did not create this instance's private model configuration.")
        cfg = _read_json(config)
        if (not prepared.get("model_id") or cfg.get("model_name") != prepared.get("model_id")
                or (not self._local and prepared.get("model_id") != self.model_info["strata_api_model"])):
            raise StrataRuntimeError("Strata's prepared model identity is inconsistent.")
        selected = (prepared.get("plan") or {}).get("model") or {}
        files = []
        source = prepared.get("source_identity")
        if self._local:
            if not self._same_source(source):
                raise StrataRuntimeError("Strata's prepared model no longer matches the chosen local files.")
            recipe = source.get("recipe") or {}
            if (selected.get("family") != recipe.get("family") or selected.get("model") != recipe.get("model")
                    or selected.get("own_files") is not True):
                raise StrataRuntimeError("Strata's launch plan substituted a different local model.")
            allowed = {Path(item["path"]).resolve() for item in source["files"]}
            argv = cfg.get("args") or []
            ggufs = [Path(value).resolve() for value in argv if isinstance(value, str) and value.lower().endswith(".gguf")]
            if any(path not in allowed for path in ggufs) or (not ggufs and recipe.get("family") != "deepseek41"):
                raise StrataRuntimeError("Strata's prepared configuration selected different model weights.")
            for index, value in enumerate(argv[:-1]):
                if value == "--pack" and not _inside(Path(argv[index + 1]), self._data_dir):
                    raise StrataRuntimeError("Strata's prepared pack is not isolated to the chosen model.")
            tokenizer = cfg.get("tokenizer")
            if tokenizer and not _inside(Path(tokenizer), self._data_dir):
                raise StrataRuntimeError("Strata's tokenizer is not isolated to the chosen model.")
            files.extend(dict(item) for item in source["files"])
        elif (selected.get("family") != self.model_info["strata_family"]
                or selected.get("model") != self.model_info["native_strata_model"]):
            raise StrataRuntimeError("Strata's launch plan selected a different model.")
        for value in cfg.get("args") or []:
            if isinstance(value, str) and Path(value).is_absolute() and _inside(Path(value), self.root) and Path(value).exists():
                candidate = Path(value)
                if candidate.is_file() and candidate.suffix.lower() == ".gguf":
                    files.append({"path": value, "size": candidate.stat().st_size})
                elif candidate.is_dir():
                    files.append({"path": value})
                    # Prepared packs are shallow. Stat their existing direct
                    # files so truncation/removal invalidates the offline hint.
                    files.extend({"path": str(p), "size": p.stat().st_size}
                                 for p in candidate.iterdir() if p.is_file())
        if not files:
            raise StrataRuntimeError("Strata did not report any prepared local model files.")
        _write_json(self.root / "prepared" / f"{self.model_key}.json", {
            "release": RELEASE, "model_key": self.model_key, "model_id": prepared["model_id"],
            "config": str(config), "files": files, "source_identity": source,
        })

    def _http_json(self, path: str) -> dict:
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=5)
        try:
            connection.request("GET", path, headers={"Authorization": f"Bearer {self._api_key}"})
            response = connection.getresponse()
            data = response.read(_MAX_JSON + 1)
            if response.status != 200 or len(data) > _MAX_JSON:
                raise StrataRuntimeError("The owned Strata service returned an invalid health response.")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise StrataRuntimeError("The owned Strata service returned invalid JSON.")
            return value
        finally:
            connection.close()

    def _validate_health(self, expected_model: str, requested_context: int) -> tuple[dict, int]:
        health, models = self._http_json("/health"), self._http_json("/v1/models")
        if (health.get("service") != "strata" or health.get("loaded") is not True
                or health.get("model") != expected_model or health.get("api_key") is not True):
            raise StrataRuntimeError("The local service is not the loaded, authenticated Strata model Studio prepared.")
        served = [m for m in models.get("data", []) if isinstance(m, dict) and m.get("id") == expected_model]
        if not served or (served[0].get("status") or {}).get("value") != "loaded":
            raise StrataRuntimeError("The prepared Strata model is not listed as loaded by the owned server.")
        context = health.get("max_context")
        if type(context) is not int or context < requested_context:
            raise StrataRuntimeError("Strata started with less context than Studio requested.")
        return health, context

    def start(self, context_tokens: int, progress=None, cancelled=None) -> dict:
        options = strata_launch_options(self.model_info, context_tokens=context_tokens,
                                        memory_limit_gb=self.speed_memory_limit_gb)
        # Reset only a previous operation's cancellation; stop() uses this same
        # mutex so a cancellation after this operation starts cannot be erased.
        with self._process_lock:
            if not self._lifecycle_lock.acquire(blocking=False):
                raise StrataRuntimeError("This Strata runtime is already changing models.")
            self._cancel.clear()
        if self._result is not None and self._result["requested_context"] != context_tokens:
            self._lifecycle_lock.release()
            raise StrataRuntimeError("Unload Strata before changing its context size.")
        try:
            if self._result is not None:
                self._check(cancelled)
                self._validate_health(self._result["model_id"], context_tokens)
                return dict(self._result)
            self._check(cancelled)
            _private_dir(self.session)
            if self._local:
                _write_json(self.session / "source.json", self._source_record())
                (self.session / "prepare-local.py").write_text(_LOCAL_SETUP_ADAPTER, encoding="utf-8")
            self._api_key = secrets.token_urlsafe(32)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", 0))
                self._port = reservation.getsockname()[1]
            self._ensure_terminal(progress, cancelled)
            launch = ["launch", *options, "--data-dir", str(self._data_dir), "--access", "local",
                      "--port", str(self._port), "--api-key-env", "SMARTAI_STRATA_API_KEY", "--tune", "skip",
                      "--save-preset", self.preset, "--no-tui", "--output", "json", "--feed", RELEASE_URL + "/" + self._feed_name()]
            self._emit(progress, "Strata is checking the chosen model and automatic GPU/RAM/SSD placement" if self._local else
                       "Strata is checking the engine, model download and automatic GPU/RAM/SSD placement")
            prepared = self._run([str(self._python()), "-c", _ADAPTER, str(self.terminal), "prepare", *launch],
                                 timeout=self.PREPARE_TIMEOUT, progress=progress, cancelled=cancelled)
            if prepared.get("ok") is not True or prepared.get("profile") != self.profile:
                raise StrataRuntimeError("Strata did not prepare the owned model profile.")
            self._remember_prepared(prepared)
            self._check(cancelled)
            if self._local and not self._same_source(prepared.get("source_identity")):
                raise StrataRuntimeError("The chosen local model changed before loading.")
            self._owner_requested, self._launch_epoch = True, time.time()
            self._emit(progress, "Loading Strata's native engine with automatic GPU/RAM/SSD placement")
            owner = self._run(self._cli("runtime", "start", "--profile", self.profile, "--die-with-pid", str(os.getpid()),
                                       "--wait", str(self.START_TIMEOUT), "--shutdown-timeout", "5", "--output", "json"),
                              timeout=self.START_TIMEOUT + 30, progress=progress, cancelled=cancelled)
            self._owner = owner
            if owner.get("ok") is not True or owner.get("state") != "ready" or owner.get("profile") != self.profile:
                raise StrataRuntimeError("Strata's owned native server did not become ready.")
            self._capture_owner()
            self._check(cancelled)
            health, context = self._validate_health(prepared["model_id"], context_tokens)
            self._check(cancelled)
            self._result = {
                "base_url": f"http://127.0.0.1:{self._port}", "api_key": self._api_key,
                "model_id": prepared["model_id"], "health": health, "n_ctx": context,
                "requested_context": context_tokens, "plan": prepared.get("plan") or self._plan,
                "profile": self.profile, "owner_pid": owner.get("host_pid"), "release": RELEASE,
                "memory_policy": strata_memory_policy(self.speed_memory_limit_gb),
                "source_identity": prepared.get("source_identity"),
            }
            self._emit(progress, "Strata is loaded; automatic GPU/RAM/SSD placement is active")
            return dict(self._result)
        except BaseException:
            self._stop_owned()
            self._result = None
            raise
        finally:
            self._lifecycle_lock.release()

    def _stop_owned(self) -> None:
        if not self._owner_requested:
            return
        args = ["runtime", "stop", "--profile", self.profile, "--wait", "15", "--output", "json"]
        if isinstance(self._owner.get("owner_id"), str):
            args += ["--if-owner-id", self._owner["owner_id"]]
        if isinstance(self._owner.get("instance_id"), str):
            args += ["--if-instance-id", self._owner["instance_id"]]
        stop_failed = False
        try:
            stopped = self._run(self._cli(*args), timeout=self.STOP_TIMEOUT, stopping=True)
            if stopped.get("ok") is not True:
                raise StrataRuntimeError("The owned Strata server did not stop normally.")
        except Exception:
            stop_failed = True
            # The official stop may be unavailable after a failed start. Only
            # identities this object has already verified can be cleaned up.
            import psutil
            self.owned_pids()
            with self._process_lock:
                identities = dict(self._owned)
            remaining = []
            for pid, started in identities.items():
                if pid == os.getpid():
                    continue
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == started:
                        process.terminate()
                        remaining.append(process)
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(remaining, timeout=3)
            for process in alive:
                try:
                    process.kill()
                except psutil.Error:
                    pass
        # An owner may return its stop acknowledgement just before it exits.
        # Never discard its identity while a verified process still holds RAM.
        import psutil
        remaining = []
        for pid in self.owned_pids():
            try:
                process = psutil.Process(pid)
                if process.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=3)
        still_running = []
        for process in alive:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    still_running.append(process)
            except psutil.NoSuchProcess:
                pass
        if still_running or (stop_failed and not remaining and
                     (self.session / "runtime" / f"{self.profile}.json").exists()):
            raise StrataRuntimeError("The owned Strata runtime could not be fully stopped; its identity is retained for another cleanup attempt.")
        self._owner_requested = False
        self._owner = {}

    def stop(self) -> None:
        """Cancel preparation immediately, then stop only this instance's server."""
        with self._process_lock:
            self._cancel.set()
            process = self._process
        self._terminate_process(process)
        with self._lifecycle_lock:
            self._stop_owned()
            self._result = None
