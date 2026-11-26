"""One bounded recovery policy for Studio's Hugging Face transfers.

Normal transfers retain the user's Hub configuration. An accelerated-transfer
failure gets one ordinary HTTP attempt. Failed files are restarted through the
Hub's public API: an older Xet partial is not necessarily a contiguous HTTP
prefix. Completed, commit-pinned cache entries are reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import importlib
import inspect
import os
from pathlib import Path
import shlex
import sys
import threading
from typing import Callable


class DownloadCancelled(RuntimeError):
    pass


_TRANSFER_LOCK = threading.RLock()
_INSTALL_LOCK = threading.RLock()
_hub = None
_original_snapshot = None
_original_file = None
_LIBRARY_PREFIXES = (
    "transformers", "diffusers", "mlx_lm", "mlx_vlm", "mlx_audio", "mflux",
    "stable_audio_3", "stable_audio_tools", "moss_soundeffect_v2",
)


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("Download cancelled; completed model files were kept.")


def _offline(kwargs=None):
    if (kwargs or {}).get("local_files_only"):
        return True
    truthy = {"1", "ON", "YES", "TRUE"}
    return any(os.environ.get(name, "").upper() in truthy
               for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OFFLINE"))


def _errors(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def _retry_kind(error):
    chain = list(_errors(error))
    text = " ".join(str(item).lower() for item in chain)
    for item in chain:
        if isinstance(item, DownloadCancelled):
            return None
        if isinstance(item, OSError) and item.errno in {
            errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS, getattr(errno, "EDQUOT", -1),
        }:
            return None
        if getattr(getattr(item, "response", None), "status_code", None) in {400, 401, 403, 404}:
            return None
        if type(item).__name__ in {
            "GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError",
            "EntryNotFoundError", "LocalEntryNotFoundError", "OfflineModeIsEnabled",
        }:
            return None
    if any(word in text for word in ("no space left", "disk full", "permission denied", "read-only file system")):
        return None
    if any(word in text for word in (
        "xet", "cas service error", "file reconstruction error", "internal writer error",
        "background writer channel closed", "hf_transfer", "hf-transfer",
    )):
        return "http"
    if any(type(item).__name__ in {
        "ConnectionError", "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutError",
        "RemoteProtocolError", "ChunkedEncodingError", "ReadError",
    } for item in chain):
        return "retry"
    return None


def format_download_error(error):
    """Keep the actual failure, adding an actionable explanation for local I/O."""
    for item in _errors(error):
        number = getattr(item, "errno", None)
        if number in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}:
            return "Not enough free disk space for the model download. Free space in the model cache and retry. " + str(item)
        if number in {errno.EACCES, errno.EPERM, errno.EROFS}:
            return "The model cache is not writable. Check its location and permissions, then retry. " + str(item)
    return str(error)


def install_hf_transfer_policy():
    """Cover public Hub calls made inside inference libraries, including aliases.

    Snapshot's own file calls stay unwrapped: its worker threads run inside the
    outer serialized snapshot operation, and failure recovery belongs to that
    outer operation. No private cache layout or download implementation is used.
    """
    global _hub, _original_snapshot, _original_file
    with _INSTALL_LOCK:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError:
            _hub, _original_snapshot, _original_file = None, None, None
            return False
        current_snapshot = getattr(hub, "snapshot_download", None)
        current_file = getattr(hub, "hf_hub_download", None)
        if _hub is not hub:
            _hub, _original_snapshot, _original_file = hub, None, None
        if callable(current_snapshot) and current_snapshot is not snapshot_download:
            _original_snapshot = current_snapshot
        if callable(current_file) and current_file is not hf_hub_download:
            _original_file = current_file
        if _original_snapshot is None and _original_file is None:
            return False
        # Import these before replacing public exports so their internal imports
        # cannot accidentally reenter a lock from a snapshot worker thread.
        for name in ("huggingface_hub._snapshot_download", "huggingface_hub.file_download"):
            try:
                importlib.import_module(name)
            except ImportError:
                pass
        if _original_snapshot is not None:
            hub.snapshot_download = snapshot_download
        if _original_file is not None:
            hub.hf_hub_download = hf_hub_download
        file_module = sys.modules.get("huggingface_hub.file_download")
        if file_module is not None and _original_file is not None:
            file_module.hf_hub_download = hf_hub_download
        snapshot_module = sys.modules.get("huggingface_hub._snapshot_download")
        if snapshot_module is not None:
            if _original_snapshot is not None:
                snapshot_module.snapshot_download = snapshot_download
            if hasattr(snapshot_module, "hf_hub_download") and _original_file is not None:
                snapshot_module.hf_hub_download = _original_file
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith(_LIBRARY_PREFIXES):
                continue
            for attr, value in list(vars(module).items()):
                if _original_file is not None and value is _original_file:
                    setattr(module, attr, hf_hub_download)
                elif _original_snapshot is not None and value is _original_snapshot:
                    setattr(module, attr, snapshot_download)
        return True


@contextmanager
def _http_transport():
    """Temporarily change both imported constants and inherited child settings."""
    constants = getattr(_hub, "constants", None)
    if constants is None:
        try:
            constants = importlib.import_module("huggingface_hub.constants")
        except ImportError:
            constants = None
    changes = {"HF_HUB_DISABLE_XET": ("1", True), "HF_HUB_ENABLE_HF_TRANSFER": ("0", False)}
    previous_env = {name: os.environ.get(name) for name in changes}
    previous_constants = {}
    try:
        for name, (value, flag) in changes.items():
            os.environ[name] = value
            if constants is not None and hasattr(constants, name):
                previous_constants[name] = getattr(constants, name)
                setattr(constants, name, flag)
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in previous_constants.items():
            setattr(constants, name, value)


def _notify(callback, message):
    if callback is not None:
        callback(message, 15.0)


def _with_cancel_progress(function, kwargs, cancel_event):
    result = dict(kwargs)
    if cancel_event is None or "tqdm_class" in result:
        return result
    try:
        parameters = inspect.signature(function).parameters
        if "tqdm_class" not in parameters:
            return result
        from tqdm.auto import tqdm
        class CancelableProgress(tqdm):
            def update(self, n=1):
                _check_cancelled(cancel_event)
                return super().update(n)
        result["tqdm_class"] = CancelableProgress
    except (ImportError, TypeError, ValueError):
        pass
    return result


def hf_hub_download(*args, cancel_event=None, progress_callback=None, **kwargs):
    """Hub single-file API with one safe retry, preserving its exceptions."""
    install_hf_transfer_policy()
    if _original_file is None:
        raise ImportError("huggingface_hub is required to download models")
    _check_cancelled(cancel_event)
    options = _with_cancel_progress(_original_file, kwargs, cancel_event)
    if _offline(options):
        # Cache hints run on the UI thread. They must never wait behind a long
        # network transfer, and no transport switch affects a local-only read.
        result = _original_file(*args, **{**options, "local_files_only": True})
        _check_cancelled(cancel_event)
        return result
    with _TRANSFER_LOCK:
        _check_cancelled(cancel_event)
        try:
            result = _original_file(*args, **options)
        except Exception as error:
            _check_cancelled(cancel_event)
            kind = None if _offline(options) or options.get("dry_run") else _retry_kind(error)
            if kind is None:
                raise
            if kind == "http":
                _notify(progress_callback, "Retrying the failed model file over HTTP; completed files are kept.")
                # Public force_download clears this file's incomplete transfer
                # under the Hub's lock, without touching other cached files.
                with _http_transport():
                    result = _original_file(*args, **{**options, "force_download": True})
            else:
                _notify(progress_callback, "Retrying the interrupted model transfer.")
                result = _original_file(*args, **options)
    _check_cancelled(cancel_event)
    return result


def _snapshot_http_retry(args, options, cancel_event):
    repo_id = options.get("repo_id") or (args[0] if args else None)
    if not repo_id or _original_file is None:
        raise RuntimeError("Cannot safely retry a snapshot without its repository and file downloader")
    from huggingface_hub.utils import filter_repo_objects
    api_fields = {key: options[key] for key in ("endpoint", "token", "headers", "library_name", "library_version", "user_agent") if key in options}
    api_parameters = inspect.signature(_hub.HfApi).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in api_parameters.values()):
        api_fields = {key: value for key, value in api_fields.items() if key in api_parameters}
    api = _hub.HfApi(**api_fields)
    info = api.repo_info(repo_id=repo_id, repo_type=options.get("repo_type"),
                         revision=options.get("revision"), token=options.get("token"))
    revision = str(getattr(info, "sha", "") or "")
    if not revision:
        raise RuntimeError("The model repository did not return an exact revision for the retry")
    requested = str(options.get("revision") or "")
    if len(requested) == 40 and all(c in "0123456789abcdefABCDEF" for c in requested) and revision.lower() != requested.lower():
        raise RuntimeError("The model repository returned a different revision during the download retry")
    names = [entry.rfilename for entry in info.siblings]
    selected = list(filter_repo_objects(names, allow_patterns=options.get("allow_patterns"),
                                        ignore_patterns=options.get("ignore_patterns")))
    snapshot_only = {"repo_id", "revision", "allow_patterns", "ignore_patterns", "max_workers", "dry_run"}
    parameters = inspect.signature(_original_file).parameters
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    common = {key: value for key, value in options.items()
              if key not in snapshot_only and (accepts_extra or key in parameters)}
    common.update(repo_id=repo_id, revision=revision)
    for filename in selected:
        _check_cancelled(cancel_event)
        # A local_dir offline fallback may return an unversioned preexisting
        # file. Only the Hub's commit-keyed cache is safe to reuse here.
        if not options.get("force_download") and not options.get("local_dir"):
            try:
                _original_file(filename=filename, **{**common, "local_files_only": True, "force_download": False})
                continue
            except Exception:
                _check_cancelled(cancel_event)
        _original_file(filename=filename, **{**common, "force_download": True, "local_files_only": False})
    _check_cancelled(cancel_event)
    # Reuse the Hub's public snapshot layout/return-value handling. Every
    # selected file is now complete, and the revision remains immutable.
    final_options = {**options, "revision": revision, "force_download": False, "local_files_only": True}
    return _original_snapshot(*args, **final_options)


def snapshot_download(*args, cancel_event=None, progress_callback=None, **kwargs):
    """Hub snapshot API; accelerated failures retry only incomplete cache files."""
    install_hf_transfer_policy()
    if _original_snapshot is None:
        raise ImportError("huggingface_hub is required to download models")
    _check_cancelled(cancel_event)
    options = _with_cancel_progress(_original_snapshot, kwargs, cancel_event)
    if _offline(options):
        result = _original_snapshot(*args, **{**options, "local_files_only": True})
        _check_cancelled(cancel_event)
        return result
    with _TRANSFER_LOCK:
        _check_cancelled(cancel_event)
        try:
            result = _original_snapshot(*args, **options)
        except Exception as error:
            _check_cancelled(cancel_event)
            kind = None if _offline(options) or options.get("dry_run") else _retry_kind(error)
            if kind is None:
                raise
            if kind == "http":
                _notify(progress_callback, "Retrying missing model files over HTTP; completed files are kept.")
                with _http_transport():
                    result = _snapshot_http_retry(args, options, cancel_event)
            else:
                _notify(progress_callback, "Retrying the interrupted model download.")
                result = _original_snapshot(*args, **options)
    _check_cancelled(cancel_event)
    return result


def run_hf_operation(operation: Callable, *, cancel_event=None, progress_callback=None):
    """Install transfer recovery before a library imports/loads a model.

    Retry happens at its Hub transfer boundary, not by repeating generation or
    keeping two partly constructed models resident in memory.
    """
    _check_cancelled(cancel_event)
    install_hf_transfer_policy()
    result = operation()
    _check_cancelled(cancel_event)
    return result


_CHILD_BOOTSTRAP = """import os, runpy, sys
helper, mode, target, *arguments = sys.argv[1:]
runpy.run_path(helper, run_name='_smartai_hf_transfer')['install_hf_transfer_policy']()
sys.argv = [target, *arguments]
if mode == 'module':
    runpy.run_module(target, run_name='__main__', alter_sys=True)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name='__main__')
"""


def hf_subprocess_command(command):
    """Bootstrap the same policy in Python trainer/console-script children.

    The child's interpreter and environment are preserved. Native executables
    are returned unchanged; their independent HTTP clients do not use Hub/Xet.
    """
    command = [str(part) for part in command]
    if not command:
        return command
    executable = command[0]
    basename = Path(executable).name.lower()
    python = basename.startswith(("python", "pypy"))
    prefix, remaining = [executable], command[1:]
    if not python:
        try:
            with open(executable, "rb") as handle:
                first = handle.readline(4096).decode("utf-8", errors="replace").strip()
        except OSError:
            return command
        if not first.startswith("#!"):
            return command
        interpreter = shlex.split(first[2:].strip())
        if not interpreter or not any(Path(part).name.lower().startswith(("python", "pypy")) for part in interpreter):
            return command
        prefix, remaining = interpreter, command
    while remaining and remaining[0] in {"-u", "-B", "-E", "-s", "-I"}:
        prefix.append(remaining.pop(0))
    if not remaining:
        return command
    if remaining[0] == "-m" and len(remaining) >= 2:
        mode, target, arguments = "module", remaining[1], remaining[2:]
    elif not remaining[0].startswith("-"):
        mode, target, arguments = "script", remaining[0], remaining[1:]
    else:
        return command
    return [*prefix, "-c", _CHILD_BOOTSTRAP, str(Path(__file__).resolve()), mode, target, *arguments]
