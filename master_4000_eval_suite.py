"""Canonical entry point for the merged 4,014-item evaluation suite.

Runs the evaluator under a small supervisor so a wedged Metal/model call cannot
leave an overnight benchmark stuck forever. The evaluator itself is unchanged.
"""
from eval._master_4000_base import *
from eval.master_4000_runtime import install

install(Master4000EvaluationEngine)

import json as _wd_json
import os as _wd_os
import subprocess as _wd_subprocess
import sys as _wd_sys
import time as _wd_time
from pathlib import Path as _WdPath

_WD_HEARTBEAT = _WdPath("eval_results/.eval_item_heartbeat.json")
_WD_STALLS = _WdPath("eval_results/.eval_item_stalls.json")
_WD_TIMEOUT = int(_wd_os.environ.get("EVAL_ITEM_WATCHDOG_SECONDS", "1200"))
_WD_MAX_STALLS = int(_wd_os.environ.get("EVAL_ITEM_MAX_STALLS", "2"))
_WD_CHILD = "EYAD_MASTER_EVAL_CHILD"


def _wd_atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_wd_json.dumps(obj), encoding="utf-8")
    _wd_os.replace(tmp, path)


def _wd_read(path):
    try:
        return _wd_json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _wd_stall_count(key):
    return int(_wd_read(_WD_STALLS).get(key, 0) or 0)


def _wd_set_stall_count(key, value):
    data = _wd_read(_WD_STALLS)
    if value <= 0:
        data.pop(key, None)
    else:
        data[key] = int(value)
    _wd_atomic_json(_WD_STALLS, data)


_original_eval_single = Master4000EvaluationEngine._evaluate_single_item


def _watchdog_eval_single(self, split_name, item):
    item_id = str(item.get("id", "unknown"))
    key = f"{split_name}:{item_id}"

    # Two genuine process-level stalls on the same item are enough: do not let one
    # pathological item consume the entire overnight run forever.
    if _wd_stall_count(key) >= _WD_MAX_STALLS:
        print(f"\n[!] Watchdog: skipping repeatedly stalled item {key}; recorded as failed.")
        _wd_set_stall_count(key, 0)
        return False

    _wd_atomic_json(
        _WD_HEARTBEAT,
        {
            "active": True,
            "pid": _wd_os.getpid(),
            "key": key,
            "split": split_name,
            "item_id": item_id,
            "started": _wd_time.time(),
        },
    )

    completed = False
    try:
        result = _original_eval_single(self, split_name, item)
        completed = True
        _wd_set_stall_count(key, 0)
        return result
    finally:
        # A hard-killed wedged child never reaches this block; the supervisor then
        # recognizes the stale active heartbeat and records the stall itself.
        if completed:
            _wd_atomic_json(
                _WD_HEARTBEAT,
                {"active": False, "pid": _wd_os.getpid(), "key": key, "ended": _wd_time.time()},
            )


Master4000EvaluationEngine._evaluate_single_item = _watchdog_eval_single


def _run_child():
    return Master4000EvaluationEngine(max_duration_hours=72.0).run_full_suite()


def _run_supervised():
    env = dict(_wd_os.environ)
    env[_WD_CHILD] = "1"

    while True:
        _wd_atomic_json(_WD_HEARTBEAT, {"active": False, "started_supervisor": _wd_time.time()})
        child = _wd_subprocess.Popen([_wd_sys.executable, "-u", _wd_os.path.abspath(__file__)], env=env)
        stalled = False

        try:
            while child.poll() is None:
                _wd_time.sleep(5.0)
                hb = _wd_read(_WD_HEARTBEAT)
                if not hb.get("active") or int(hb.get("pid", -1)) != child.pid:
                    continue

                age = _wd_time.time() - float(hb.get("started", _wd_time.time()))
                if age < _WD_TIMEOUT:
                    continue

                key = str(hb.get("key", "unknown"))
                attempt = _wd_stall_count(key) + 1
                _wd_set_stall_count(key, attempt)
                print(
                    f"\n[!] Watchdog: {key} made no item-level completion for "
                    f"{int(age)}s; killing wedged child (stall {attempt}/{_WD_MAX_STALLS}) and resuming checkpoint."
                )
                child.kill()
                child.wait()
                stalled = True
                break
        except KeyboardInterrupt:
            try:
                child.send_signal(2)
                child.wait(timeout=10)
            except Exception:
                try:
                    child.kill()
                except Exception:
                    pass
            return 130

        if stalled:
            _wd_time.sleep(2.0)
            continue

        rc = child.returncode
        if rc == 0:
            return 0
        return int(rc if rc is not None else 1)


if __name__ == "__main__":
    if _wd_os.environ.get(_WD_CHILD) == "1":
        _run_child()
    else:
        raise SystemExit(_run_supervised())
