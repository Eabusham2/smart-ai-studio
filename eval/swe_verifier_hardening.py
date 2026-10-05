"""Remove false-negative DeepSWE patch-application failures.

The compatibility sandbox historically hardcoded both a 2-second patch timeout
and `patch -p1`. The timeout could reject a correct patch on a busy machine, and
`-p1` rejects perfectly valid model diffs whose headers are `app/file.py` rather
than `a/app/file.py` / `b/app/file.py`.

This layer derives a bounded setup timeout from the configured sandbox timeout,
selects the correct strip level for the emitted diff, dry-runs before applying,
and leaves the actual test command under the original sandbox resource policy.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _extract_patch(text: str) -> str:
    blocks = re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```", text or "", re.I)
    return (blocks[-1] if blocks else (text or "")).strip() + "\n"


def _patch_strip_level(patch: str) -> int:
    """Use -p1 for git a/... b/... headers, otherwise preserve plain paths with -p0."""
    paths = []
    for match in re.finditer(r"^(?:---|\+\+\+)\s+([^\t\r\n ]+)", patch or "", re.M):
        path = match.group(1).strip()
        if path and path != "/dev/null":
            paths.append(path)
    if paths and all(path.startswith(("a/", "b/")) for path in paths):
        return 1
    return 0


def _install_one(cls, result_cls, *, use_git_apply: bool = False) -> None:
    if getattr(cls, "_smartai_patch_timeout_hardened", False):
        return

    def verify_git_diff_patch(self, repo_structure, patch_text, test_cmd):
        root = tempfile.mkdtemp(prefix="smartai_swe_patch_")
        started = time.perf_counter()
        try:
            max_files = int(getattr(self, "max_files", 64) or 64)
            if len(repo_structure) > max_files:
                return result_cls(False, 0.0, "", "too many files", 0.0)

            network_guard = getattr(self, "_network_guard", None)
            if callable(network_guard):
                network_guard(root)

            for rel, body in repo_structure.items():
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    return result_cls(False, 0.0, "", "unsafe path", 0.0)
                dest = Path(root) / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(str(body), encoding="utf-8")

            patch = _extract_patch(patch_text)
            (Path(root) / "task.patch").write_text(patch, encoding="utf-8")

            configured = float(getattr(self, "timeout", 4.0) or 4.0)
            patch_timeout = max(4.0, configured)

            # Windows does not normally ship the POSIX `patch` utility.
            # Keep the historical Unix path when present, otherwise use Git's
            # cross-platform unified-diff parser.
            git_bin = shutil.which("git")
            patch_bin = shutil.which("patch")
            use_git = bool(
                use_git_apply
                or platform.system() == "Windows"
                or not patch_bin
            )

            if use_git:
                if not git_bin:
                    return result_cls(
                        False,
                        (time.perf_counter() - started) * 1000.0,
                        "",
                        "SWE patch verification requires Git (or POSIX patch on Unix).",
                        0.0,
                    )
                subprocess.run(
                    [git_bin, "init", "-q"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=patch_timeout,
                )
                check = subprocess.run(
                    [git_bin, "apply", "--check", "task.patch"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=patch_timeout,
                )
                if check.returncode:
                    return result_cls(
                        False,
                        (time.perf_counter() - started) * 1000.0,
                        check.stdout,
                        check.stderr[-2000:],
                        0.0,
                    )
                applied = subprocess.run(
                    [git_bin, "apply", "task.patch"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=patch_timeout,
                )
            else:
                strip = _patch_strip_level(patch)
                check = subprocess.run(
                    [patch_bin, "--dry-run", f"-p{strip}", "-i", "task.patch"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=patch_timeout,
                )
                if check.returncode:
                    return result_cls(
                        False,
                        (time.perf_counter() - started) * 1000.0,
                        check.stdout,
                        check.stderr[-2000:],
                        0.0,
                    )
                applied = subprocess.run(
                    [patch_bin, f"-p{strip}", "-i", "task.patch"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=patch_timeout,
                )

            if applied.returncode:
                return result_cls(
                    False, (time.perf_counter() - started) * 1000.0,
                    applied.stdout, applied.stderr[-2000:], 0.0,
                )

            runner = getattr(self, "_run", None)
            if callable(runner):
                return runner(test_cmd, root, shell=True)

            env = dict(os.environ)
            env["PYTHONPATH"] = root + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            preexec = None
            limits = getattr(self, "_limits", None)
            if callable(limits) and platform.system() != "Windows":
                preexec = limits
            proc = subprocess.run(
                test_cmd,
                shell=True,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=configured,
                preexec_fn=preexec,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            if proc.returncode == 0:
                return result_cls(True, elapsed, proc.stdout, None, 1.0)
            return result_cls(False, elapsed, proc.stdout, proc.stderr[-2000:], 0.0)

        except subprocess.TimeoutExpired as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            cmd = getattr(exc, "cmd", "patch/test command")
            return result_cls(False, elapsed, exc.stdout or "", f"Timeout: {cmd}", 0.0)
        except Exception as exc:
            return result_cls(
                False,
                (time.perf_counter() - started) * 1000.0,
                "",
                f"{type(exc).__name__}: {exc}",
                0.0,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    cls.verify_git_diff_patch = verify_git_diff_patch
    cls._smartai_patch_timeout_hardened = True


def install(compat_module) -> None:
    """Patch the benchmark compatibility sandbox and core SWE sandbox if present."""
    _install_one(
        compat_module.POSIXHardenedSandbox,
        compat_module.SandboxResult,
        use_git_apply=False,
    )

    try:
        from core.swe_sandbox import HardenedSWESandbox, SandboxResult
    except Exception:
        return

    _install_one(HardenedSWESandbox, SandboxResult, use_git_apply=True)
