"""Replace a staged desktop package after shutdown, keeping data and a backup.

The helper is an OS process, not an interpreter inside the tree being replaced.
This matters on Windows, where an open Python executable prevents replacement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid


BUILD_RECORD = ".smartai-build.json"
PENDING_RECORD = ".smartai-update-pending.json"
REPOSITORY = "Eabusham2/smart-ai-studio"
_LAYOUTS = {".", "Contents/Resources/app", "SmartAI.AppDir/usr/bin"}
_USER_NAMES = {
    "data", "preloaded_models", "models", "cache", "caches", ".cache", ".env",
    ".venv", "venv", "logs", "outputs", "exports", "artifacts", "workspaces",
    "custom_models.json", "memory.db",
}
_IGNORED_NAMES = {".git", "__pycache__", PENDING_RECORD}


def _record(code: Path) -> dict:
    path = code / BUILD_RECORD
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise RuntimeError("This installation is missing its release build identity.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema") != 1
            or value.get("repository", value.get("repo")) != REPOSITORY
            or not re.fullmatch(r"[0-9a-fA-F]{40}", str(value.get("commit", "")))):
        raise RuntimeError("The release build identity is invalid.")
    return value


def _names(record: dict, key: str) -> set[str]:
    result = record.get(key, [])
    if not isinstance(result, list) or any(
        not isinstance(name, str) or name in {"", ".", ".."}
        or any(char in name for char in "/\\:\x00") for name in result
    ):
        raise RuntimeError("The package file manifest is invalid.")
    return set(result)


def preservation_paths(install: Path, staged: Path, code_relative: str) -> list[str]:
    """Find user-owned children without following their links or copying weights."""
    old_code, new_code = install / code_relative, staged / code_relative
    new_record = _record(new_code)
    try:
        old_record = _record(old_code)
    except (OSError, ValueError, RuntimeError):
        # Migration from packages predating the updater. The backup also keeps
        # anything which cannot be confidently classified as user content.
        old_record = new_record
    selected = set()
    roots = [(old_code, "managed_paths")]
    if old_code != install:
        roots.append((install, "install_managed_paths"))
    for root, key in roots:
        managed = _names(old_record, key) or _names(new_record, key)
        if not managed:
            raise RuntimeError("The release is missing its managed file list.")
        for child in root.iterdir():
            if child.name in _IGNORED_NAMES:
                continue
            if child.name in _USER_NAMES or child.name not in managed:
                selected.add(child.relative_to(install).as_posix())
    custom = os.environ.get("SMARTAI_DATA_DIR")
    if custom:
        data = Path(custom).expanduser().resolve()
        if data.is_relative_to(install) and old_code.is_relative_to(data):
            raise RuntimeError("The data directory contains the application code. Choose a separate data folder before updating.")
        if data.is_relative_to(install) and data.exists():
            # Do not carry an old code/runtime folder over the freshly staged
            # package merely because it was configured as a data directory.
            relative = data.relative_to(install)
            is_user_folder = (data.is_relative_to(old_code)
                              and data.relative_to(old_code).parts[0] in _USER_NAMES)
            if (staged / relative).exists() and not is_user_folder:
                raise RuntimeError("The data directory overlaps packaged application files. Choose a separate data folder before updating.")
            selected.add(data.relative_to(install).as_posix())
    # A preserved parent already contains its children.
    result = []
    for name in sorted(selected, key=lambda item: (len(PurePosixPath(item).parts), item)):
        path = PurePosixPath(name)
        if not any(path.is_relative_to(PurePosixPath(parent)) for parent in result):
            result.append(name)
    return result


def _layout(install_root, staged_root, code_relative):
    if code_relative not in _LAYOUTS:
        raise RuntimeError("Unsupported application package layout.")
    original, incoming = Path(install_root).absolute(), Path(staged_root).absolute()
    if original.is_symlink() or incoming.is_symlink():
        raise RuntimeError("Update the actual application folder, not a symbolic link.")
    install, staged = original.resolve(), incoming.resolve()
    if (install in {Path(install.anchor), Path.home().resolve()}
            or not install.is_dir() or not staged.is_dir()
            or install == staged or staged.is_relative_to(install)
            or install.is_relative_to(staged)):
        raise RuntimeError("Invalid application update destination.")
    if (install / ".git").exists():
        raise RuntimeError("A Git checkout must be updated through Git; use a packaged installation for in-app updates.")
    if install.stat().st_dev != staged.stat().st_dev:
        raise RuntimeError("Stage the update on the same drive as the application.")
    if not os.access(install.parent, os.W_OK):
        raise RuntimeError("The application folder is read-only. Move it to a writable folder before updating.")
    for root in (install, staged):
        code = root / code_relative
        if not code.resolve().is_relative_to(root):
            raise RuntimeError("Application code escapes its package folder.")
        if not (code / "bootstrap_runtime.py").is_file():
            raise RuntimeError("The package is missing its application launcher.")
        python = code / "runtime/python" / ("python.exe" if os.name == "nt" else "bin/python3")
        if not python.is_file() or not python.resolve().is_relative_to(root):
            raise RuntimeError("The package is missing its bundled Python runtime.")
    _record(staged / code_relative)
    return install, staged


def _shell_script(plan: dict) -> str:
    q = shlex.quote
    install, staged, backup = (plan[key] for key in ("install", "staged", "backup"))
    transaction = plan["transaction"]
    lines = ["#!/bin/sh", "set -eu", "umask 077"]
    for key in ("install", "staged", "backup", "transaction", "python", "bootstrap", "ack", "token"):
        lines.append(f"{key}={q(plan[key])}")
    lines.extend([
        "started=0", "replaced=0", "committed=0", "child=", "rollback_running=0",
        "report() { printf '{\"status\":\"%s\"}\\n' \"$1\" > \"$transaction/result.json\"; }",
        "exists() { [ -e \"$1\" ] || [ -L \"$1\" ]; }",
        "stop_child() {",
        "  if [ -n \"$child\" ]; then",
        # The bootstrap starts a private process group, including its GUI child.
        "    kill -TERM \"-$child\" 2>/dev/null || true",
        "    sleep 1",
        "    kill -KILL \"-$child\" 2>/dev/null || true",
        "    wait \"$child\" 2>/dev/null || true",
        "  fi",
        "}",
        "launch() {",
        "  (cd \"$(dirname \"$bootstrap\")\" && exec \"$python\" -c 'import os,runpy,sys; os.setsid(); script=sys.argv.pop(1); runpy.run_path(script, run_name=\"__main__\")' \"$bootstrap\") >> \"$transaction/launch.log\" 2>&1 &",
        "  child=$!",
        "}",
        "rollback() {",
        "  [ \"$rollback_running\" = 0 ] || return",
        "  rollback_running=1",
        "  trap - EXIT HUP INT TERM",
        "  set +e",
        "  stop_child",
        "  if [ \"$replaced\" = 1 ] && exists \"$install\"; then",
    ])
    for index, relative in reversed(list(enumerate(plan["preserve"]))):
        old = str(Path(backup) / relative)
        new = str(Path(install) / relative)
        marker = str(Path(transaction) / "moves" / str(index))
        lines.extend([
            f"    if [ -f {q(marker)} ] && ! exists {q(old)} && exists {q(new)}; then",
            f"      mkdir -p {q(str(Path(old).parent))}",
            f"      mv {q(new)} {q(old)} || {{ report recovery_needed; exit 1; }}",
            "    fi",
        ])
    lines.extend([
        "    mv \"$install\" \"$transaction/failed-package\" || { report recovery_needed; exit 1; }",
        "  fi",
        "  if [ \"$started\" = 1 ] && exists \"$backup\" && ! exists \"$install\"; then",
        "    mv \"$backup\" \"$install\" || { report recovery_needed; exit 1; }",
        "    launch",
        "  elif [ \"$started\" = 1 ] && [ \"$replaced\" = 0 ] && exists \"$install\" && ! exists \"$backup\"; then",
        "    launch",
        "  fi",
        "  report failed",
        "}",
        "trap 'if [ \"$committed\" = 0 ]; then rollback; fi' EXIT",
        "trap 'exit 1' HUP INT TERM",
        "report waiting",
    ])
    for pid in plan["wait_pids"]:
        lines.extend([
            "ticks=0",
            f"while kill -0 {pid} 2>/dev/null; do",
            f"  [ \"$ticks\" -lt {plan['shutdown_timeout']} ] || {{ echo 'Application did not exit; update cancelled.'; exit 1; }}",
            "  sleep 1; ticks=$((ticks + 1))",
            "done",
        ])
    lines.extend([
        "report installing",
        "started=1", "mv \"$install\" \"$backup\"",
        "replaced=1", "mv \"$staged\" \"$install\"",
        "mkdir -p \"$transaction/moves\" \"$transaction/displaced\"",
    ])
    for index, relative in enumerate(plan["preserve"]):
        old, new = str(Path(backup) / relative), str(Path(install) / relative)
        held = str(Path(transaction) / "displaced" / str(index))
        marker = str(Path(transaction) / "moves" / str(index))
        lines.extend([
            f"if exists {q(old)}; then",
            f"  if exists {q(new)}; then mv {q(new)} {q(held)}; fi",
            f"  mkdir -p {q(str(Path(new).parent))}",
            f"  : > {q(marker)}",
            f"  mv {q(old)} {q(new)}",
            "fi",
        ])
    lines.extend([
        "report restarting", "launch", "ticks=0",
        f"while [ \"$ticks\" -lt {plan['startup_timeout']} ]; do",
        "  if [ -f \"$ack\" ] && [ \"$(cat \"$ack\")\" = \"$token\" ]; then",
        "    committed=1; report complete; exit 0",
        "  fi",
        "  kill -0 \"$child\" 2>/dev/null || { echo 'Updated application exited before becoming ready.'; exit 1; }",
        "  sleep 1; ticks=$((ticks + 1))",
        "done",
        "echo 'Updated application did not become ready; restoring the previous version.'",
        "exit 1",
    ])
    return "\n".join(lines) + "\n"


_POWERSHELL = r'''param([Parameter(Mandatory=$true)][string]$PlanPath)
$ErrorActionPreference = 'Stop'
$p = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$started = $false
$replaced = $false
$committed = $false
$child = $null
function Report([string]$status) {
    @{status=$status} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $p.transaction 'result.json') -Encoding UTF8
}
function Move-Retry([string]$source, [string]$destination) {
    for ($attempt=0; $attempt -lt 20; $attempt++) {
        try { Move-Item -LiteralPath $source -Destination $destination -ErrorAction Stop; return }
        catch { if ($attempt -eq 19) { throw }; Start-Sleep -Milliseconds 500 }
    }
}
function Launch-App {
    $arguments = '"' + $p.bootstrap + '"'
    Start-Process -FilePath $p.python -ArgumentList $arguments -WorkingDirectory (Split-Path -Parent $p.bootstrap) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $p.transaction 'launch.log') -RedirectStandardError (Join-Path $p.transaction 'launch-error.log')
}
function Stop-Child {
    if ($null -ne $script:child) {
        # Windows PowerShell 5.1 can turn native stderr into a terminating
        # error under Stop. A process which already exited must never prevent
        # the subsequent data-restoration transaction.
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $script:child.Refresh()
            if (-not $script:child.HasExited) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $script:child.Id /T /F 2>$null | Out-Null
                $script:child.WaitForExit(10000) | Out-Null
            }
        } catch {
            # Move-Retry below still detects any remaining file locks.
        } finally {
            $ErrorActionPreference = $previousPreference
        }
    }
}
try {
    Report 'waiting'
    foreach ($processId in $p.wait_pids) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -ne $process -and -not $process.WaitForExit([int]$p.shutdown_timeout * 1000)) {
            throw 'Application did not exit; update cancelled.'
        }
    }
    Report 'installing'
    $started = $true
    Move-Retry $p.install $p.backup
    $replaced = $true
    Move-Retry $p.staged $p.install
    New-Item -ItemType Directory -Path (Join-Path $p.transaction 'moves'), (Join-Path $p.transaction 'displaced') -Force | Out-Null
    for ($i=0; $i -lt $p.preserve.Count; $i++) {
        $source = Join-Path $p.backup $p.preserve[$i]
        $destination = Join-Path $p.install $p.preserve[$i]
        if (Test-Path -LiteralPath $source) {
            if (Test-Path -LiteralPath $destination) {
                Move-Retry $destination (Join-Path (Join-Path $p.transaction 'displaced') ([string]$i))
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Set-Content -LiteralPath (Join-Path (Join-Path $p.transaction 'moves') ([string]$i)) -Value ''
            Move-Retry $source $destination
        }
    }
    Report 'restarting'
    $child = Launch-App
    for ($tick=0; $tick -lt $p.startup_timeout; $tick++) {
        if ((Test-Path -LiteralPath $p.ack) -and ((Get-Content -LiteralPath $p.ack -Raw).Trim() -ceq $p.token)) {
            $committed = $true
            Report 'complete'
            exit 0
        }
        if ($child.HasExited) { throw 'Updated application exited before becoming ready.' }
        Start-Sleep -Seconds 1
        $child.Refresh()
    }
    throw 'Updated application did not become ready.'
} catch {
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $p.transaction 'error.log')
    try {
        Stop-Child
        if ($replaced -and (Test-Path -LiteralPath $p.install)) {
            for ($i=$p.preserve.Count-1; $i -ge 0; $i--) {
                $source = Join-Path $p.backup $p.preserve[$i]
                $destination = Join-Path $p.install $p.preserve[$i]
                $marker = Join-Path (Join-Path $p.transaction 'moves') ([string]$i)
                if ((Test-Path -LiteralPath $marker) -and -not (Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $destination)) {
                    New-Item -ItemType Directory -Path (Split-Path -Parent $source) -Force | Out-Null
                    Move-Retry $destination $source
                }
            }
            Move-Retry $p.install (Join-Path $p.transaction 'failed-package')
        }
        if ($started -and (Test-Path -LiteralPath $p.backup) -and -not (Test-Path -LiteralPath $p.install)) {
            Move-Retry $p.backup $p.install
            $child = Launch-App
        } elseif ($started -and -not $replaced -and (Test-Path -LiteralPath $p.install) -and -not (Test-Path -LiteralPath $p.backup)) {
            # A residual runtime lock can prevent the very first rename. The
            # original app is still intact, so reopen it after the failed update.
            $child = Launch-App
        }
        Report 'failed'
    } catch {
        $_ | Out-String | Add-Content -LiteralPath (Join-Path $p.transaction 'error.log')
        Report 'recovery_needed'
    }
    exit 1
}
'''


def schedule_install(install_root: Path, staged_root: Path, code_relative: str, *,
                     gui_pid: int | None = None, bootstrap_pid: int | None = None):
    """Launch a detached helper; the caller must then use normal app shutdown.

    Until the specified processes exit, the helper can be terminated to cancel
    the handoff. Nothing inside the active installation is changed beforehand.
    """
    install, staged = _layout(install_root, staged_root, code_relative)
    preserve = preservation_paths(install, staged, code_relative)
    for relative in preserve:
        for root in (install, staged):
            if not (root / relative).parent.resolve().is_relative_to(root):
                raise RuntimeError("A preserved data path has a parent outside the application.")
    transaction = Path(tempfile.mkdtemp(prefix=".smartai-update-", dir=install.parent))
    token = uuid.uuid4().hex
    code = install / code_relative
    runtime_python = code / "runtime/python" / ("python.exe" if os.name == "nt" else "bin/python3")
    if bootstrap_pid is None:
        candidate = os.environ.get("SMARTAI_BOOTSTRAP_PID", "")
        bootstrap_pid = int(candidate) if candidate.isdecimal() else None
    pids = [os.getpid() if gui_pid is None else gui_pid, bootstrap_pid]
    if any(pid is not None and (not isinstance(pid, int) or pid < 0) for pid in pids):
        raise RuntimeError("Invalid application process identity.")
    plan = {
        "install": str(install), "staged": str(staged),
        "backup": str(install.with_name(install.name + ".previous-" + token[:8])),
        "transaction": str(transaction), "python": str(runtime_python),
        "bootstrap": str(code / "bootstrap_runtime.py"),
        "preserve": preserve, "wait_pids": sorted({pid for pid in pids if pid}),
        "shutdown_timeout": 300, "startup_timeout": 1800,
        "ack": str(transaction / ("ready-" + token)), "token": token,
    }
    pending = {"ack_path": plan["ack"], "token": token,
               "commit": _record(staged / code_relative)["commit"]}
    pending_path = staged / code_relative / PENDING_RECORD
    pending_path.write_text(json.dumps(pending) + "\n", encoding="utf-8")
    manifest = transaction / "plan.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    log = transaction / "helper.log"
    try:
        if os.name == "nt":
            helper = transaction / "install.ps1"
            helper.write_text(_POWERSHELL, encoding="utf-8-sig")
            powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            command = [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                       "-File", str(helper), "-PlanPath", str(manifest)]
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        else:
            helper = transaction / "install.sh"
            helper.write_text(_shell_script(plan), encoding="utf-8")
            helper.chmod(0o700)
            command = ["/bin/sh", str(helper)]
            options = {"start_new_session": True}
        with log.open("ab") as output:
            env = dict(os.environ)
            if env.get("SMARTAI_DATA_DIR"):
                env["SMARTAI_DATA_DIR"] = str(Path(env["SMARTAI_DATA_DIR"]).expanduser().resolve())
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, cwd=transaction, env=env, **options)
    except Exception:
        pending_path.unlink(missing_ok=True)
        shutil.rmtree(transaction)
        raise
    process.smartai_update_directory = transaction
    process.smartai_update_backup = Path(plan["backup"])
    process.smartai_update_log = log
    # Do not close the GUI merely because Popen succeeded. In particular,
    # PowerShell policy/startup failures otherwise strand a user with no app.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            status = json.loads((transaction / "result.json").read_text(encoding="utf-8-sig")).get("status")
        except (OSError, ValueError, AttributeError):
            status = None
        if status in {"waiting", "installing", "restarting", "complete"} and process.poll() in {None, 0}:
            return process
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    pending_path.unlink(missing_ok=True)
    raise RuntimeError(f"The update helper could not start. The application is still open. Details: {log}")


def acknowledge_startup(app_root: Path | None = None) -> bool:
    """Called once the new GUI is ready, allowing the helper to retain its backup."""
    code = Path(app_root or Path(__file__).resolve().parents[1]).resolve()
    pending = code / PENDING_RECORD
    if not pending.is_file():
        return False
    try:
        if pending.stat().st_size > 4096:
            return False
        value = json.loads(pending.read_text(encoding="utf-8"))
        token = value.get("token", "")
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            return False
        if value.get("commit") != _record(code)["commit"]:
            return False
        ack = Path(value.get("ack_path", ""))
        if (not ack.is_absolute() or ack.name != "ready-" + token
                or not ack.parent.name.startswith(".smartai-update-")
                or ack.is_symlink() or ack.parent.is_symlink() or not ack.parent.is_dir()):
            return False
        # The transaction must be beside this installed package, never an
        # arbitrary path supplied by a file in the application directory.
        if ack.parent.parent not in code.parents:
            return False
        try:
            with ack.open("x", encoding="utf-8") as handle:
                handle.write(token + "\n")
        except FileExistsError:
            if ack.read_text(encoding="utf-8").strip() != token:
                return False
        pending.unlink(missing_ok=True)
        return True
    except (OSError, ValueError, TypeError, RuntimeError):
        return False
