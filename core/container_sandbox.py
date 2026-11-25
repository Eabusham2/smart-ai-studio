"""Mandatory OS-container boundary for generated code execution.

The helpers in this module never execute candidate-controlled code on the host.
Docker is deliberately fail-closed: an unavailable image or daemon is reported
to the caller instead of falling back to a local interpreter or shell.
"""
from __future__ import annotations

import base64
import io
import os
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_SANDBOX_IMAGE = "python:3.10-slim"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 32 * 1024 * 1024

_ALLOWED_ENVIRONMENT = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "TMPDIR": "/tmp",
}

_PYTHON_BOOTSTRAP = """import os, sys
os.environ.clear()
os.environ.update({environment!r})
source = sys.stdin.buffer.read()
exec(compile(source, "<generated verifier>", "exec"), {{"__name__": "__main__"}})
""".format(environment=_ALLOWED_ENVIRONMENT)

_WORKSPACE_BOOTSTRAP = """import base64, os, pathlib, subprocess, sys, tarfile
os.environ.clear()
allowed = {environment!r}
allowed["PYTHONPATH"] = "/workspace"
os.environ.update(allowed)
command = base64.b64decode(sys.argv[1], validate=True).decode("utf-8")
with tarfile.open(fileobj=sys.stdin.buffer, mode="r:gz") as archive:
    members = archive.getmembers()
    total = 0
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not member.isfile():
            raise ValueError("unsafe workspace archive member")
        total += member.size
        if total > {workspace_limit}:
            raise ValueError("workspace archive exceeds size limit")
    archive.extractall("/workspace", members=members)
completed = subprocess.run(command, cwd="/workspace", env=allowed, shell=True)
raise SystemExit(completed.returncode)
""".format(environment=_ALLOWED_ENVIRONMENT, workspace_limit=MAX_WORKSPACE_BYTES)


@dataclass(frozen=True)
class ContainerExecution:
    returncode: Optional[int]
    stdout: str
    stderr: str
    execution_time_ms: float
    timed_out: bool = False
    output_limited: bool = False
    error: Optional[str] = None


def build_docker_command(
    image: str,
    cidfile: str,
    max_memory_mb: int,
    *,
    workspace_mb: int = 64,
    container_name: Optional[str] = None,
) -> list[str]:
    """Build the invariant container policy without any host filesystem mount."""
    selected_image = str(image or "").strip()
    if not selected_image:
        raise ValueError("A sandbox image is required")
    memory_mb = max(64, int(max_memory_mb))
    workspace_mb = max(16, min(int(workspace_mb), memory_mb))
    command = [
        "docker", "run", "--rm", "--interactive", "--init", "--pull", "never",
        "--cidfile", cidfile,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", "64",
        "--memory", f"{memory_mb}m",
        "--memory-swap", f"{memory_mb}m",
        "--cpus", "1.0",
        "--ulimit", "core=0:0",
        "--ulimit", "nofile=64:64",
        "--user", "65534:65534",
        "--ipc", "none",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777,uid=65534,gid=65534",
        "--tmpfs", f"/workspace:rw,noexec,nosuid,nodev,size={workspace_mb}m,mode=700,uid=65534,gid=65534",
        "--workdir", "/workspace",
        "--env", "HOME=/tmp",
        "--env", "LANG=C.UTF-8",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONIOENCODING=utf-8",
        "--env", "PYTHONNOUSERSITE=1",
        "--entrypoint", "python",
    ]
    if container_name:
        command.extend(["--name", container_name])
    command.append(selected_image)
    return command


def _bounded_reader(stream, output: bytearray, limit: int, exceeded: threading.Event) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            remaining = limit - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                exceeded.set()
    finally:
        stream.close()


def _input_writer(stream, payload: bytes) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        stream.close()


def _container_id(cidfile: str) -> Optional[str]:
    try:
        value = Path(cidfile).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if value and len(value) <= 128 and all(character in "0123456789abcdefABCDEF" for character in value):
        return value
    return None


def _force_remove_container(cidfile: str, container_name: Optional[str] = None) -> None:
    target = _container_id(cidfile) or container_name
    if not target:
        return
    try:
        subprocess.run(
            ["docker", "rm", "--force", target],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _stop_process(
    process: subprocess.Popen, cidfile: str, container_name: Optional[str] = None
) -> None:
    try:
        # Give the attached Docker client a chance to proxy termination to PID 1.
        process.terminate()
    except OSError:
        pass
    _force_remove_container(cidfile, container_name)
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def run_container(
    image: str,
    arguments: list[str],
    payload: bytes,
    *,
    timeout: float,
    max_memory_mb: int,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    workspace_mb: int = 64,
) -> ContainerExecution:
    """Run one bounded container and remove its complete process tree on failure."""
    started = time.perf_counter()
    if len(payload) > MAX_INPUT_BYTES:
        return ContainerExecution(
            None, "", "", 0.0, error=f"Sandbox input exceeds {MAX_INPUT_BYTES} bytes"
        )

    with tempfile.TemporaryDirectory(prefix="smartai_container_") as control_dir:
        cidfile = os.path.join(control_dir, "container.cid")
        container_name = "smartai-" + Path(control_dir).name.replace("_", "-")
        try:
            command = build_docker_command(
                image,
                cidfile,
                max_memory_mb,
                workspace_mb=workspace_mb,
                container_name=container_name,
            ) + list(arguments)
            options = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "bufsize": 0,
            }
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(command, **options)
        except (OSError, ValueError) as error:
            elapsed = (time.perf_counter() - started) * 1000.0
            return ContainerExecution(None, "", "", elapsed, error=str(error))

        stdout = bytearray()
        stderr = bytearray()
        output_exceeded = threading.Event()
        readers = [
            threading.Thread(
                target=_bounded_reader,
                args=(process.stdout, stdout, max_output_bytes, output_exceeded),
                daemon=True,
            ),
            threading.Thread(
                target=_bounded_reader,
                args=(process.stderr, stderr, max_output_bytes, output_exceeded),
                daemon=True,
            ),
        ]
        writer = threading.Thread(
            target=_input_writer, args=(process.stdin, payload), daemon=True
        )
        for thread in readers:
            thread.start()
        writer.start()

        timed_out = False
        output_limited = False
        deadline = time.monotonic() + max(0.1, float(timeout))
        while process.poll() is None:
            if output_exceeded.is_set():
                output_limited = True
                _stop_process(process, cidfile, container_name)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _stop_process(process, cidfile, container_name)
                break
            time.sleep(0.01)

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _stop_process(process, cidfile, container_name)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        for thread in readers:
            thread.join(timeout=1.0)
        _force_remove_container(cidfile, container_name)

        elapsed = (time.perf_counter() - started) * 1000.0
        return ContainerExecution(
            process.returncode,
            bytes(stdout).decode("utf-8", errors="replace"),
            bytes(stderr).decode("utf-8", errors="replace"),
            elapsed,
            timed_out=timed_out,
            output_limited=output_limited or output_exceeded.is_set(),
        )


def run_python_in_container(
    source: str,
    *,
    timeout: float,
    max_memory_mb: int,
    image: str = DEFAULT_SANDBOX_IMAGE,
) -> ContainerExecution:
    payload = str(source).encode("utf-8")
    return run_container(
        image,
        ["-I", "-c", _PYTHON_BOOTSTRAP],
        payload,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
    )


def _workspace_archive(root: str, *, max_files: int) -> bytes:
    buffer = io.BytesIO()
    count = 0
    total = 0
    root_path = Path(root)
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root_path.rglob("*")):
            relative = path.relative_to(root_path)
            if not relative.parts or relative.parts[0] == ".git" or relative.name == "task.patch":
                continue
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Workspace contains unsupported file type: {relative}")
            count += 1
            total += metadata.st_size
            if count > max_files:
                raise ValueError("Workspace contains too many files")
            if total > MAX_WORKSPACE_BYTES:
                raise ValueError("Workspace exceeds sandbox size limit")
            info = tarfile.TarInfo(relative.as_posix())
            info.size = metadata.st_size
            info.mode = 0o755 if metadata.st_mode & stat.S_IXUSR else 0o644
            info.uid = info.gid = 65534
            info.uname = info.gname = "sandbox"
            with path.open("rb") as source:
                archive.addfile(info, source)
    return buffer.getvalue()


def run_workspace_in_container(
    root: str,
    command: str,
    *,
    timeout: float,
    max_memory_mb: int,
    max_files: int,
    image: str = DEFAULT_SANDBOX_IMAGE,
) -> ContainerExecution:
    if not isinstance(command, str) or not command.strip():
        return ContainerExecution(None, "", "", 0.0, error="A test command is required")
    if len(command.encode("utf-8")) > 16384:
        return ContainerExecution(None, "", "", 0.0, error="Test command exceeds size limit")
    try:
        payload = _workspace_archive(root, max_files=max_files)
    except (OSError, ValueError, tarfile.TarError) as error:
        return ContainerExecution(None, "", "", 0.0, error=str(error))
    encoded_command = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return run_container(
        image,
        ["-I", "-c", _WORKSPACE_BOOTSTRAP, encoded_command],
        payload,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        workspace_mb=64,
    )
