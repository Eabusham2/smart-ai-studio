"""Mandatory container sandbox used by generated-code and benchmark tasks."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

from core.container_sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    MAX_WORKSPACE_BYTES,
    ContainerExecution,
    run_python_in_container,
    run_workspace_in_container,
)


@dataclass
class SandboxResult:
    passed: bool
    execution_time_ms: float
    output: str
    error: Optional[str] = None
    reward: float = 0.0


class HardenedSWESandbox:
    """Run candidate code only inside the shared fail-closed OS container."""

    _smartai_patch_timeout_hardened = True

    def __init__(
        self,
        timeout_sec: float = 4.0,
        max_memory_mb: int = 512,
        max_files: int = 64,
        docker_image: str = DEFAULT_SANDBOX_IMAGE,
    ):
        self.timeout = float(timeout_sec)
        self.max_memory_mb = max(64, int(max_memory_mb))
        self.max_files = int(max_files)
        self.docker_image = str(docker_image or DEFAULT_SANDBOX_IMAGE)

    @staticmethod
    def _result(execution: ContainerExecution) -> SandboxResult:
        passed = (
            execution.returncode == 0
            and not execution.error
            and not execution.timed_out
            and not execution.output_limited
        )
        if execution.error:
            error = f"Sandbox unavailable: {execution.error}"
        elif execution.timed_out:
            error = "ProcessTimeout: container removed"
        elif execution.output_limited:
            error = "OutputLimitExceeded: container removed"
        elif execution.returncode:
            error = execution.stderr[-2000:] or f"Container exited with {execution.returncode}"
        else:
            error = None
        return SandboxResult(
            passed,
            execution.execution_time_ms,
            execution.stdout,
            error,
            1.0 if passed else 0.0,
        )

    def execute_python_code(self, code: str, tests: str) -> SandboxResult:
        source = (
            "# -*- coding: utf-8 -*-\n"
            "import collections, itertools, json, math\n\n"
            f"{code}\n\n# UNIT TESTS\n{tests}\n"
        )
        return self._result(
            run_python_in_container(
                source,
                timeout=self.timeout,
                max_memory_mb=self.max_memory_mb,
                image=self.docker_image,
            )
        )

    def _run(self, command, cwd: str, shell: bool = False) -> SandboxResult:
        """Compatibility runner that keeps legacy callers inside the container."""
        if isinstance(command, str):
            command_text = command
        else:
            command_text = shlex.join(str(part) for part in command)
        return self._result(
            run_workspace_in_container(
                cwd,
                command_text,
                timeout=self.timeout,
                max_memory_mb=self.max_memory_mb,
                max_files=self.max_files,
                image=self.docker_image,
            )
        )

    @staticmethod
    def _safe_relative_path(value: str) -> Optional[PurePosixPath]:
        raw = str(value or "")
        normalized = raw.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or "\x00" in normalized
            or normalized.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized)
            or ".." in path.parts
            or (path.parts and path.parts[0] == ".git")
        ):
            return None
        return path

    def verify_git_diff_patch(
        self,
        repo_structure: Dict[str, str],
        patch_text: str,
        test_cmd: str,
    ) -> SandboxResult:
        root = tempfile.mkdtemp(prefix="smartai_swe_stage_")
        started = time.perf_counter()
        try:
            if len(repo_structure) > self.max_files:
                return SandboxResult(False, 0.0, "", "too many files", 0.0)
            repository_bytes = 0
            for relative, body in repo_structure.items():
                safe_path = self._safe_relative_path(relative)
                if safe_path is None:
                    return SandboxResult(False, 0.0, "", "unsafe path", 0.0)
                encoded_body = str(body).encode("utf-8")
                repository_bytes += len(encoded_body)
                if repository_bytes > MAX_WORKSPACE_BYTES:
                    return SandboxResult(False, 0.0, "", "repository exceeds size limit", 0.0)
                destination = Path(root).joinpath(*safe_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded_body)

            blocks = re.findall(
                r"```(?:diff|patch)?\s*([\s\S]*?)```", patch_text or "", re.I
            )
            patch = (blocks[-1] if blocks else (patch_text or "")).strip() + "\n"
            if len(patch.encode("utf-8")) > 2 * 1024 * 1024:
                return SandboxResult(False, 0.0, "", "patch exceeds size limit", 0.0)
            Path(root, "task.patch").write_text(patch, encoding="utf-8")

            git = shutil.which("git")
            if not git:
                return SandboxResult(
                    False,
                    (time.perf_counter() - started) * 1000.0,
                    "",
                    "SWE patch staging requires Git.",
                    0.0,
                )
            patch_timeout = max(4.0, self.timeout)
            git_environment = {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "HOME": root,
                "PATH": os.environ.get("PATH", ""),
            }
            subprocess.run(
                [git, "init", "-q"],
                cwd=root,
                env=git_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=patch_timeout,
                check=True,
            )
            check = subprocess.run(
                [git, "apply", "--check", "task.patch"],
                cwd=root,
                env=git_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=patch_timeout,
                check=False,
            )
            if check.returncode:
                return SandboxResult(
                    False,
                    (time.perf_counter() - started) * 1000.0,
                    check.stdout.decode("utf-8", errors="replace"),
                    check.stderr.decode("utf-8", errors="replace")[-2000:],
                    0.0,
                )
            applied = subprocess.run(
                [git, "apply", "task.patch"],
                cwd=root,
                env=git_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=patch_timeout,
                check=False,
            )
            if applied.returncode:
                return SandboxResult(
                    False,
                    (time.perf_counter() - started) * 1000.0,
                    applied.stdout.decode("utf-8", errors="replace"),
                    applied.stderr.decode("utf-8", errors="replace")[-2000:],
                    0.0,
                )
            return self._result(
                run_workspace_in_container(
                    root,
                    test_cmd,
                    timeout=self.timeout,
                    max_memory_mb=self.max_memory_mb,
                    max_files=self.max_files,
                    image=self.docker_image,
                )
            )
        except subprocess.TimeoutExpired as error:
            return SandboxResult(
                False,
                (time.perf_counter() - started) * 1000.0,
                "",
                f"Patch staging timeout: {error.cmd}",
                0.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return SandboxResult(
                False,
                (time.perf_counter() - started) * 1000.0,
                "",
                f"Patch staging failed: {error}",
                0.0,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def evaluate_dsl_expression(expr):
        match = re.match(r"\s*\[([^\]]*)\]", expr)
        if not match:
            return None
        try:
            values = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
        except Exception:
            return None
        remaining = expr[match.end():]
        while remaining.strip():
            remaining = remaining.lstrip()
            match = re.match(r">>~fold\((\d+)\)", remaining)
            if match:
                if values:
                    shift = int(match.group(1)) % len(values)
                    values = values[shift:] + values[:shift]
                remaining = remaining[match.end():]
                continue
            match = re.match(r"<#>scale\((-?\d+)\)", remaining)
            if match:
                scale = int(match.group(1))
                values = [value * scale for value in values]
                remaining = remaining[match.end():]
                continue
            match = re.match(r"@fuse\s*\[([^\]]*)\]", remaining)
            if match:
                try:
                    right = [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
                except Exception:
                    return None
                if len(right) != len(values):
                    return None
                values = [left + right_value for left, right_value in zip(values, right)]
                remaining = remaining[match.end():]
                continue
            return None
        return values

    def create_local_worktree(self, repo_dir):
        destination = tempfile.mkdtemp(prefix="smartai_worktree_")
        shutil.rmtree(destination)
        try:
            completed = subprocess.run(
                ["git", "worktree", "add", "--detach", destination, "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=8,
            )
            return destination if completed.returncode == 0 else None
        except Exception:
            return None

    def cleanup_local_worktree(self, repo_dir, path):
        if path:
            subprocess.run(
                ["git", "worktree", "remove", "--force", path],
                cwd=repo_dir,
                capture_output=True,
            )
            shutil.rmtree(path, ignore_errors=True)
