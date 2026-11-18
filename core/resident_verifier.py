"""Supervise an isolated checked Python task and verify its per-process peak resident bytes."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


_OUTPUT_BYTES = 64 * 1024  # A checked reward harness prints a timing number and short diagnostics.


def _candidate():
    import resource

    for kind, requested in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, _OUTPUT_BYTES)):
        finite = [value for value in resource.getrlimit(kind) if value != resource.RLIM_INFINITY]
        cap = min([requested, *finite])
        resource.setrlimit(kind, (cap, cap))
    try:
        exec(compile(sys.stdin.buffer.read(), "<checked task>", "exec"), {"__name__": "__main__"})
    except BaseException as exc:
        print(f"Verifier task failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


def _supervise(requested, psutil_directory, poll_seconds):
    import resource
    # -I -S excludes site initialization from both interpreters. Only the
    # trusted supervisor imports the copy of psutil already used by Studio.
    sys.path.append(psutil_directory)
    import psutil

    inherited = [value for kind in (resource.RLIMIT_DATA, resource.RLIMIT_AS)
                 for value in resource.getrlimit(kind) if value != resource.RLIM_INFINITY]
    cap = min([requested, *inherited])
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    # Set a flag rather than raising while Popen is constructing the child.
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-I", "-S", __file__, "candidate"]
    # Fork/exec from this small interpreter so the worker's RSS high-water
    # mark does not start at the footprint of the serving model's process.
    with subprocess.Popen(command) as child:
        try:
            process = None
            while True:
                if stopping:
                    return 1
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    peak = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
                    if peak <= 0:
                        raise RuntimeError("kernel peak resident memory unavailable")
                    if peak > cap:
                        print(f"Verifier task failed: peak resident memory {peak} exceeded {cap} bytes", file=sys.stderr)
                        return 1
                    if child.returncode < 0:
                        print(f"Verifier task failed: signal {-child.returncode}", file=sys.stderr)
                    return 0 if child.returncode == 0 else 1
                try:
                    if process is None:
                        process = psutil.Process(child.pid)
                    rss = process.memory_info().rss
                except psutil.NoSuchProcess:
                    # A short-lived child can exit before inspection. wait4
                    # provides its own trusted final peak and reaps this PID.
                    continue
                if rss > cap:
                    print(f"Verifier task failed: resident memory {rss} exceeded {cap} bytes", file=sys.stderr)
                    return 1
                time.sleep(poll_seconds)
        finally:
            if child.returncode is None:
                # Popen.poll/kill can reap before wait4 obtains this child's
                # usage. Signal its unreaped PID directly and consume its status.
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, _ = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)


def run_resident_task(program, byte_budget, timeout, *, poll_seconds=.01):
    import psutil
    command = [sys.executable, "-I", "-S", __file__, "supervise", str(byte_budget),
               str(Path(psutil.__file__).resolve().parent.parent), str(poll_seconds)]
    # Files avoid pipe deadlock on large case sets and bound what the host
    # reads from the checked worker. Its file-size limit bounds output writes.
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        source.write(program.encode("utf-8"))
        source.seek(0)
        with subprocess.Popen(command, stdin=source, stdout=out, stderr=err, start_new_session=True) as supervisor:
            try:
                status = supervisor.wait(timeout=timeout)
            finally:
                if supervisor.returncode is None:
                    try:
                        supervisor.terminate()
                        supervisor.wait(timeout=max(1., poll_seconds + .5))
                    except BaseException:
                        # Also clean up if the supervisor itself is stuck or a
                        # second interrupt arrives during graceful teardown.
                        try:
                            os.killpg(supervisor.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        supervisor.wait()
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(command, status, out.read(_OUTPUT_BYTES).decode("utf-8", "replace"),
                                            err.read(_OUTPUT_BYTES).decode("utf-8", "replace"))


if __name__ == "__main__":
    if sys.argv[1] == "candidate":
        sys.exit(_candidate())
    try:
        sys.exit(_supervise(int(sys.argv[2]), sys.argv[3], float(sys.argv[4])))
    except Exception as exc:
        print(f"Verifier memory setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
